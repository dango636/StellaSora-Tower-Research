"""CPython MT19937 and latent-state packing shared by CPU preparation and CUDA.

Keep this code independent of torch: background input preparation never owns a
CUDA context. The seed order and the number/order of Random calls are unchanged.
"""
from __future__ import annotations

from random import Random
import numpy as np
from .vendor import stella_sim as sim


def pack_streams(state, seeds):
    randoms = np.empty((len(seeds), 625), dtype=np.uint32)
    latents = np.empty((len(seeds), 5), dtype=np.int32)
    for i, seed in enumerate(seeds):
        rng = Random(int(seed))
        latent = sim.sample_latent(state, rng)
        latents[i] = latent.gE, latent.fE, latent.gC, latent.fC, latent.G1
        randoms[i] = rng.getstate()[1]
    return randoms, latents
