"""GPU trajectories with seeded exploration and public-only root selection."""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
import time

import numpy as np
import torch

from .features import PHASES, KINDS
from .sampling_descriptors import HARD_KIND_NAMES
from .gpu_game_tables import LAYOUT, MAX_ACTIONS, pack_state, unpack_state
from .gpu_games import GpuGames, device, pack_streams
from .presets import Preset
from .sampling import build_sampled_family, select_family_roots, root_selection_role
from .vendor import stella_sim as sim


@dataclass(frozen=True, slots=True)
class _RootPosition:
    phase: str
    floor: int
    gap: float
    frame: int
    row: int
    hard_kind: str | None = None
    teacher_action: int = -1
    student_action: int = -1
    disagreement_gap: float = 0.0


class GpuFamilySampler:
    """Keep complete trajectories on CUDA and unpack only selected roots."""

    def __init__(self, model_paths: dict[str, str]):
        self.games = {"teacher": GpuGames(model_paths["teacher"])}
        if "behavior" in model_paths:
            self.games["behavior"] = (self.games["teacher"]
                if model_paths["behavior"] == model_paths["teacher"] else GpuGames(model_paths["behavior"]))
        self.teacher = self.games["teacher"]
        self._model_paths = dict(model_paths)
        self._behavior_game = self.games.get("behavior")
        if self._behavior_game is self.teacher:
            self._behavior_game = None
        self._behavior_path = model_paths.get("behavior") if self._behavior_game is not None else None
        self.information = {key: value for key, value in self.teacher.information.items()
                            if key != "model_parameter_sha256"}
        self.information.update(
            models={name: game.information["model_parameter_sha256"]
                    for name, game in self.games.items()},
            exploration_rng="CPython Random per family",
            root_selection="natural / phase-floor / student-teacher disagreement when enabled / hard-category fallback",
            posterior_serialization="exact Fraction(float), normalized",
        )
        self.stats = dict(families=0, roots=0, batches=0, gap_action_rows=0,
                          comparison_action_rows=0, sampling_seconds=0.0)
        self._action_columns = torch.arange(MAX_ACTIONS, device="cuda")
        self._score_weights = device(np.asarray([min(i//6 % 7, i % 6+1) for i in range(252)],
                                                dtype=np.int32))

    def set_models(self, model_paths, sources):
        """Reuse the teacher and student CUDA slots across all preset groups."""
        if model_paths["teacher"] != self._model_paths["teacher"]:
            self.teacher.set_model(sources["teacher"])
        if "behavior" in model_paths:
            if model_paths["behavior"] == model_paths["teacher"]:
                self.games["behavior"] = self.teacher
            else:
                if self._behavior_game is None:
                    self._behavior_game = GpuGames(sources["behavior"])
                elif model_paths["behavior"] != self._behavior_path:
                    self._behavior_game.set_model(sources["behavior"])
                self._behavior_path = model_paths["behavior"]
                self.games["behavior"] = self._behavior_game
        self._model_paths = dict(model_paths)
        self.information = dict(self.information, models={
            name: game.information["model_parameter_sha256"] for name, game in self.games.items()})

    def _resident_games(self):
        return [self.teacher] + ([self._behavior_game] if self._behavior_game is not None else [])

    def _gaps(self, values, counts, offsets):
        if not values.numel():
            return torch.zeros_like(counts, dtype=torch.float32)
        sizes = torch.where(counts > 1, counts, 0)
        indexes = (offsets-sizes)[:, None]+self._action_columns
        valid = self._action_columns < sizes[:, None]
        safe = torch.where(valid, indexes, 0)
        q = values.gather(0, safe.reshape(-1)).reshape_as(safe).masked_fill(~valid, -torch.inf)
        best = q.topk(2, dim=1).values
        return torch.where(counts > 1, best[:, 0]-best[:, 1], 0.0)

    def _policy_comparison(self, teacher_values, student_values, counts, offsets):
        if teacher_values.numel() != student_values.numel():
            raise ValueError("Teacher/student compact action rows must have identical length")
        if not teacher_values.numel():
            return torch.zeros((len(counts), 2), dtype=torch.int64, device=counts.device), torch.zeros_like(counts, dtype=torch.float32)
        sizes = torch.where(counts > 1, counts, 0)
        indexes = (offsets-sizes)[:, None]+self._action_columns
        valid = self._action_columns < sizes[:, None]
        indexes = torch.where(valid, indexes, 0)
        teacher_q = teacher_values.gather(0, indexes.reshape(-1)).reshape_as(indexes).masked_fill(~valid, -torch.inf)
        student_q = (teacher_q if student_values is teacher_values else
                     student_values.gather(0, indexes.reshape(-1)).reshape_as(indexes).masked_fill(~valid, -torch.inf))
        teacher_choice, student_choice = teacher_q.argmax(1), student_q.argmax(1)
        margin = (teacher_q.gather(1, teacher_choice[:, None])-
                  teacher_q.gather(1, student_choice[:, None])).squeeze(1)
        return torch.stack((teacher_choice, student_choice), dim=1), torch.where(counts > 1, margin, 0.0)

    def _sample_group(self, tasks, presets, game, control):
        n = len(tasks)
        packed, beliefs, randoms = [], [], []
        for group, (task, preset) in enumerate(zip(tasks, presets)):
            initial = sim.initial_from_pools(preset.pools)
            state, belief = pack_state(initial, group)
            rng, latent = pack_streams(initial, [task["environment_seed"]])
            state[LAYOUT["gE"]:LAYOUT["gE"]+5] = latent[0]
            packed.append(state); beliefs.append(belief); randoms.append(rng[0])
        exploration = [Random(task["exploration_seed"]) for task in tasks]
        state_frames, belief_frames, metadata, gap_frames = [], [], [], []
        hard_frames = []
        hard_enabled = any(task.get("hard_sampling") and any(
            root_selection_role(task, j) == "hard" for j in range(task["roots_per_family"]))
            for task in tasks)
        disagreement_enabled = any(task.get("disagreement_sampling") for task in tasks)
        gap_enabled = any(root_selection_role(task, j) == "hard"
                          for task in tasks for j in range(task["roots_per_family"]))
        teacher_scores_needed = hard_enabled or disagreement_enabled or gap_enabled
        exploration_enabled = any(task["epsilon"] > 0 for task in tasks)
        decision_frames, disagreement_gap_frames = [], []

        def observe(states, posterior, _randoms, _actions, counts, offsets, features, choices, values):
            state_frames.append(states.clone())
            belief_frames.append(posterior.clone())
            scores = (states[:, LAYOUT["H"]:LAYOUT["H"]+252]*self._score_weights).sum(1)
            eligible = (counts > 1) & (scores < states[:, LAYOUT["cap"]])
            metadata.append(torch.stack((states[:, LAYOUT["steps"]], states[:, LAYOUT["phase"]],
                                         states[:, LAYOUT["floor"]], eligible.to(torch.int32)), dim=1))
            actor_values = values
            if teacher_scores_needed and game is not self.teacher and values.numel():
                values = self.teacher._values(features[:values.numel()])
                self.stats["gap_action_rows"] += values.numel()
            gap_frames.append(self._gaps(values, counts, offsets) if gap_enabled else
                              torch.zeros_like(counts, dtype=torch.float32))
            teacher_choices = choices if game is self.teacher else None
            if disagreement_enabled:
                student = self.games["behavior"]
                if student is game:
                    student_values = actor_values
                elif student is self.teacher or not values.numel():
                    student_values = values
                else:
                    student_values = student._values(features[:values.numel()])
                    self.stats["comparison_action_rows"] += values.numel()
                decisions, margins = self._policy_comparison(values, student_values, counts, offsets)
                decision_frames.append(decisions)
                disagreement_gap_frames.append(margins)
                teacher_choices = decisions[:, 0]
            if hard_enabled:
                if teacher_choices is None:
                    teacher_choices = self._policy_comparison(values, values, counts, offsets)[0][:, 0]
                # One small category readback at trajectory completion replaces
                # five synchronous CPU transfers on every decision step. A
                # single kernel avoids dozens of tiny PyTorch elementwise
                # launches for the many small, distinct teacher groups.
                teacher_choices = teacher_choices.to(dtype=torch.int32)
                categories = torch.empty_like(counts, dtype=torch.int32)
                self.teacher.module.launch("game_hard_category", n,
                    (states, _actions, counts, teacher_choices, categories, n))
                hard_frames.append(categories)
            if exploration_enabled:
                overrides = np.full(n, -1, dtype=np.int32)
                for i, count in enumerate(counts.cpu().numpy()):
                    # Keep each nonzero-epsilon family's exact CPython draw
                    # schedule, including draws on single-action states.
                    if count and exploration[i].random() < tasks[i]["epsilon"]:
                        choice = exploration[i].randrange(int(count))
                        if count > 1:
                            overrides[i] = choice
                if np.any(overrides >= 0):
                    replacement = device(overrides)
                    choices.copy_(torch.where(replacement >= 0, replacement, choices))

        scores, steps = game._run(np.asarray(packed), np.asarray(beliefs), np.asarray(randoms),
                                 list(range(n+1)), control=control, observe=observe)
        meta = torch.stack(metadata).cpu().numpy()
        gaps = torch.stack(gap_frames).cpu().numpy()
        hard_categories = torch.stack(hard_frames).cpu().numpy() if hard_enabled else None
        if disagreement_enabled:
            decisions = torch.stack(decision_frames).cpu().numpy()
            disagreement_gaps = torch.stack(disagreement_gap_frames).cpu().numpy()
        selected = []
        for row, task in enumerate(tasks):
            if control:
                control()
            eligible = [(int(step), _RootPosition(PHASES[int(phase)], int(floor),
                                                 float(gaps[frame, row]), frame, row,
                                                 HARD_KIND_NAMES[int(hard_categories[frame, row])]
                                                 if hard_enabled and task.get("hard_sampling") else None,
                                                 int(decisions[frame, row, 0]) if disagreement_enabled else -1,
                                                 int(decisions[frame, row, 1]) if disagreement_enabled else -1,
                                                 float(disagreement_gaps[frame, row]) if disagreement_enabled else 0.0))
                        for frame, (step, phase, floor, keep) in enumerate(meta[:, row]) if keep]
            selected.append(select_family_roots(task, eligible, lambda state: state.gap, lambda state: state.hard_kind,
                disagreement=(lambda state: state.disagreement_gap if state.teacher_action != state.student_action else None)
                if disagreement_enabled else None))
        positions = [position for family in selected for _, position in family]
        root_states = torch.stack([state_frames[p.frame][p.row] for p in positions]).cpu().numpy()
        root_beliefs = torch.stack([belief_frames[p.frame][p.row] for p in positions]).cpu().numpy()
        result, offset = [], 0
        for i, (task, preset, roots) in enumerate(zip(tasks, presets, selected)):
            public = [(step, unpack_state(root_states[offset+j], root_beliefs[offset+j]))
                      for j, (step, _) in enumerate(roots)]
            offset += len(roots)
            if task.get("hard_sampling"):
                task = dict(task, _hard_kinds={step: position.hard_kind for j, (step, position) in enumerate(roots)
                                             if root_selection_role(task, j) == "hard"})
            if disagreement_enabled:
                task = dict(task, _policy_decisions={step: {
                    "teacher_action_index": position.teacher_action, "student_action_index": position.student_action,
                    "teacher_student_disagreement": position.teacher_action != position.student_action,
                    "teacher_student_predicted_gap": position.disagreement_gap,
                } for step, position in roots})
            result.append(build_sampled_family(task, preset, public, int(steps[i]), int(scores[i]),
                                                self.information))
        return result

    @torch.inference_mode()
    def sample_families(self, tasks, control=None):
        tasks = list(tasks)
        if not tasks:
            return []
        started = time.perf_counter()
        presets = [Preset.from_dict(task["preset"]) for task in tasks]
        groups = {name: [] for name in self.games}
        for i, task in enumerate(tasks):
            name = ("behavior" if task.get("behavior_source") in ("last_candidate", "best_candidate")
                    and "behavior" in self.games else "teacher")
            groups[name].append(i)
        results = [None]*len(tasks)
        for name, indexes in groups.items():
            if indexes:
                rows = self._sample_group([tasks[i] for i in indexes], [presets[i] for i in indexes],
                                          self.games[name], control)
                for i, row in zip(indexes, rows):
                    results[i] = row
        self.stats["families"] += len(results)
        self.stats["roots"] += sum(len(row["roots"]) for row in results)
        self.stats["batches"] += 1
        self.stats["sampling_seconds"] += time.perf_counter()-started
        return results

    def snapshot(self):
        totals = {key: sum(game.stats[key] for game in self._resident_games())
                  for key in ("games", "decisions", "action_rows", "seconds",
                              "host_input_validation_calls", "host_input_validation_seconds",
                              "host_input_validation_failed_calls", "host_input_validation_failed_seconds")}
        return dict(self.information, **self.stats, **totals)

    def close(self):
        self._action_columns = self._score_weights = None
        for game in self._resident_games():
            game.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
