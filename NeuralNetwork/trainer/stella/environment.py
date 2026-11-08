"""A small public-information interface around the frozen game kernel.

Only this module owns live environment randomness and hidden counters. Policies
receive PublicState and legal actions. Counterfactual branches resample hidden
counters from that public state's exact posterior and use independent RNGs.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from random import Random
from typing import Callable

from .vendor import stella_sim as sim
from .presets import Preset, canonical_seed, validate_role_totals

PublicState = sim.PublicState
Action = sim.Action
ModelGap = sim.ModelGap
score = sim.score
action_key = sim.action_key
MAX_DECISIONS = 1600


@lru_cache(maxsize=2048)
def original_role_totals(inventory, pools):
    """Recover original targets from acquired items AND remaining pools."""
    totals = [sum(pools[2*r]) + sum(pools[2*r+1]) for r in range(3)]
    for role, _quality, target, _level, count in inventory:
        totals[role-1] += target*count
    return tuple(totals)


def validate_state_preset(state: PublicState, preset: Preset) -> None:
    """Check the ORIGINAL complete target vector, then retained state counts."""
    # Constructing Preset validates all 36 original slots and the scope.
    if not isinstance(preset, Preset):
        raise TypeError("A validated complete Preset is required")
    sim.validate_state(state)
    if state.cap != preset.total:
        raise ValueError("State cap differs from the complete original preset")
    for i, pool in enumerate(preset.pools):
        role, quality = i // 2 + 1, i % 2 + 1
        carried = Counter(state.pools[i])
        for r, q, target, _level, count in state.H:
            if r == role and q == quality:
                carried[target] += count
        if carried != Counter(pool):
            raise ValueError("State target inventory differs from its original preset")


def legal_actions(state: PublicState) -> tuple[Action, ...]:
    """Modeled legal actions; no value threshold or baseline-action lock.

    The reference simulator deliberately leaves negative-wallet event outcomes
    undefined. These are the sole additional exclusions: all positive-mass
    outcomes must retain a nonnegative wallet. This is a rule-domain check,
    not an action-value heuristic or teacher protection mechanism.
    """
    actions = sim.legal_actions(state)
    if state.phase == "EVENT":
        actions = tuple(a for a in actions if a.kind == "SKIP_EVENT" or all(
            n == 0 or state.coins + delta >= 0
            for n, _d, delta, _kind, _scope in sim.EVENT_DATA[state.eventId][2][a.index][1]))
    if not actions and state.phase != "END":
        raise ModelGap("NO_COVERED_ACTION", state.eventId)
    return actions


class Episode:
    """A mutable episode controller; its public state objects are immutable."""
    __slots__ = ("preset", "state", "steps", "reward", "_rng", "_latent", "_actions_cache")

    def __init__(self, preset: Preset, seed: int):
        if not isinstance(preset, Preset):
            raise TypeError("Episode requires a complete Preset")
        self.preset, self.steps, self.reward = preset, 0, 0
        self._actions_cache = None
        self._rng = Random(int(seed))
        initial = sim.initial_from_pools(preset.pools)
        transition = sim.advance_to_decision(initial, self._rng)
        self.state, self._latent = transition.state, transition.latent

    @classmethod
    def from_public(cls, state: PublicState, seed: int, preset: Preset | None = None) -> "Episode":
        if not isinstance(state, PublicState):
            raise TypeError("Branches accept PublicState only; never Transition/Latent")
        if preset is not None:
            validate_state_preset(state, preset)
        elif not 72 <= state.cap <= 144:
            raise ValueError("Branch state is outside the supported target range")
        validate_role_totals(original_role_totals(state.H, state.pools))
        obj = cls.__new__(cls)
        obj.preset, obj.steps, obj.reward = preset, 0, 0
        obj._actions_cache = None
        obj._rng = Random(int(seed))
        transition = sim.advance_to_decision(state, obj._rng, latent=None)
        obj.state, obj._latent = transition.state, transition.latent
        return obj

    @property
    def done(self) -> bool:
        return self.state.phase == "END"

    @property
    def score(self) -> int:
        return sim.score(self.state)

    @property
    def actions(self) -> tuple[Action, ...]:
        # Policy inference and step validation share the same immutable public
        # state. Cache only that one tuple, retaining every legality check and
        # avoiding an expensive PublicState/Fraction hash. Identity detects a
        # changed state even when an external caller explicitly replaces it.
        cached = self._actions_cache
        if cached is None or cached[0] is not self.state:
            cached = self._actions_cache = (self.state, legal_actions(self.state))
        return cached[1]

    def step(self, action: Action) -> int:
        if self.done:
            raise RuntimeError("Cannot step a finished episode")
        if self.steps >= MAX_DECISIONS:
            raise RuntimeError("Episode decision budget exceeded; no truncated reward is emitted")
        if action not in self.actions:
            raise ValueError("Action is illegal or outside modeled event outcomes")
        transition = sim.step(self.state, action, self._rng, self._latent,
                              collapse_forced=True, validate_action=False)
        self.state, self._latent = transition.state, transition.latent
        self.steps += 1
        self.reward += transition.reward
        return transition.reward

    def clone(self) -> "Episode":
        """Exact ENVIRONMENT copy for debugging; do not pass it to a policy."""
        obj = self.__class__.__new__(self.__class__)
        obj.preset, obj.state = self.preset, self.state
        obj.steps, obj.reward = self.steps, self.reward
        obj._latent = self._latent
        obj._actions_cache = self._actions_cache
        obj._rng = Random()
        obj._rng.setstate(self._rng.getstate())
        return obj


def initial(preset: Preset, seed: int) -> Episode:
    return Episode(preset, seed)


def branch(state: PublicState, action: Action, seed: int,
           preset: Preset | None = None) -> Episode:
    """One action, then a live continuation from a fresh public posterior.

    Using the same seed for competing root actions implements common random
    numbers without consulting the actual episode's private hidden state.
    """
    episode = Episode.from_public(state, seed, preset)
    episode.step(action)
    return episode


def heuristic_action(state: PublicState, actions: tuple[Action, ...] | None = None,
                     rng: Random | None = None) -> Action:
    """Deterministic cold-start policy; it is never an inference-time guard.

    Greedy ordinary gain on visible cards, free enhancements first, cheap
    resource purchase otherwise; event values use only the public rule table.
    It is deliberately labeled a heuristic, not a recovered historical model.
    """
    actions = legal_actions(state) if actions is None else actions
    if not actions:
        raise ValueError("No action in a terminal state")
    if state.phase in ("PAGE", "ENHANCE_PAGE"):
        choices = [a for a in actions if a.kind == "SELECT"]
        def utility(action):
            new, _role, _quality, target, old, level, _index = state.offer[action.index-1]
            gain = min(target, level) - min(target, old)
            # A zero-valued potential can remain a future offer distractor.
            # Small tie-breaking terms do not block learned-policy decisions.
            return gain, int(target > 0), max(target-level, 0) * int(new), -action.index
        return max(choices, key=utility)
    if state.phase == "SHOP":
        enhances = [a for a in actions if a.kind == "ENHANCE"]
        if enhances and enhances[0].price == 0:
            return enhances[0]
        buys = [a for a in actions if a.kind == "BUY"]
        if buys:
            return min(buys, key=lambda a: (a.price, a.type))
        if enhances:
            return enhances[0]
        return next(a for a in actions if a.kind == "LEAVE")
    if state.phase == "EVENT":
        def utility(action):
            if action.kind == "SKIP_EVENT":
                return 0.0
            outcomes = sim.EVENT_DATA[state.eventId][2][action.index][1]
            return sum(n/d * (delta/100 + int(kind != "none"))
                       for n, d, delta, kind, _scope in outcomes)
        return max(actions, key=utility)
    if len(actions) == 1:
        return actions[0]
    raise ValueError(f"Unexpected decision phase: {state.phase}")


def play(preset: Preset, seed: int, policy: Callable = heuristic_action,
         *, on_step: Callable | None = None) -> dict:
    """Run an entire game with an isolated planning stream and ordinary score."""
    episode = Episode(preset, seed)
    planning_rng = Random(canonical_seed("policy", seed))
    while not episode.done:
        state, actions = episode.state, episode.actions
        action = policy(state, actions, planning_rng)
        if on_step is not None:
            on_step(state, action)
        episode.step(action)
    return {"score": episode.score, "steps": episode.steps, "preset": preset.name,
            "total": preset.total, "seed": seed}
