"""Mine previous training mistakes with their original frozen teacher on CUDA."""
from __future__ import annotations

import json
import numpy as np
import torch

from .environment import legal_actions, sim, validate_state_preset
from .features import TARGET_SCALE
from .fit_torch import TorchQ
from .hard_examples import classify_hard_state
from .network import QNetwork
from .presets import Preset


class GpuHardMiner:
    def __init__(self, model_path):
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        source = QNetwork.load(model_path, frozen=True)
        self.model = TorchQ(source, "cuda").eval().requires_grad_(False)
        self.information = dict(backend="torch_cuda", device_name=torch.cuda.get_device_name(),
                                rules_version=sim.MODEL_VERSION,
                                precision="FP32", tf32=False, model_parameter_sha256=source.checksum())
        self.stats = dict(roots=0, mistakes=0)

    @torch.inference_mode()
    def mine(self, tasks, control=None):
        examples = []
        for task in tasks:
            if control:
                control()
            with np.load(task["path"], allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"].item()))
                if meta["split"] != "train":
                    raise ValueError("Hard-example mining accepts training families only")
                state = sim.state_from_dict(meta["state"])
                validate_state_preset(state, Preset.from_dict(meta["preset"]))
                examples.append((meta, state, data["x"], data["q"], data["returns"]))
        values = (self.model(torch.from_numpy(np.concatenate([e[2] for e in examples])).to("cuda"),
                             amp=False)*TARGET_SCALE).cpu().numpy()
        rows, offset = [], 0
        for meta, state, x, q, returns in examples:
            predicted = values[offset:offset+len(q)]
            offset += len(q)
            chosen, best = int(predicted.argmax()), int(q.argmax())
            delta = returns[best].astype(np.float64)-returns[chosen].astype(np.float64)
            regret, se = float(q[best]-q[chosen]), float(delta.std(ddof=1)/np.sqrt(len(delta)))
            effective = max(0.0, regret-se)
            rows.append(dict(root_id=meta["root_id"], family_id=meta["family_id"], split="train",
                             teacher_id=meta["teacher_id"],
                             category=classify_hard_state(state, legal_actions(state), chosen),
                             chosen_index=chosen, label_best_index=best, decision_regret=regret,
                             decision_regret_se=se, priority_signal=effective,
                             source_root=meta if effective > 0 else None))
            self.stats["mistakes"] += int(effective > 0)
        self.stats["roots"] += len(rows)
        return rows

    def snapshot(self):
        return dict(self.information, **self.stats)

    def close(self):
        self.model = None
        torch.cuda.empty_cache()
