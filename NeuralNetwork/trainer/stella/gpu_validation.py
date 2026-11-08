"""Validate packed rollout inputs once on the host, before raw CUDA launches.

Packed state/count fields retain their int32 ABI. Linear device addresses use
64-bit arithmetic; compact action prefixes remain bounded by INT32_MAX. This
module deliberately imports no torch and never synchronizes a CUDA stream.
"""
from __future__ import annotations

import numpy as np

from .features import KINDS, PHASES
from .gpu_game_tables import LAYOUT, MAX_ACTIONS, STATE_SIZE
from .indexing import require_integer
from .vendor import stella_sim as sim

INT32_MAX = 2**31 - 1
MAX_BATCH_GAMES = INT32_MAX // MAX_ACTIONS
ACTION_PHASES = {KINDS.index(kind): tuple(PHASES.index(phase) for phase in phases)
    for kind, phases in {
        "SELECT": ("PAGE", "ENHANCE_PAGE"), "REROLL": ("PAGE",),
        "BUY": ("SHOP",), "ENHANCE": ("SHOP",), "RESTOCK": ("SHOP",), "LEAVE": ("SHOP",),
        "EVENT_OPTION": ("EVENT",), "SKIP_EVENT": ("EVENT",), "ADVANCE": ("ADVANCE",),
        "CONFIRM_CORE": ("CORE",), "GENERATE_PAGE": ("DRAW",),
        "GENERATE_EVENT": ("EVENT_DRAW",), "OPEN_SHOP": ("SHOP_UNSEEN",),
    }.items()}


def validate_batch_size(n):
    """Bound the largest possible int32 compact action prefix without clamping."""
    return require_integer(n, "rollout batch size", maximum=MAX_BATCH_GAMES)


def _integers(value, shape, dtype, name, *, contiguous=True):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array with shape {shape}, got {array.shape}/{array.dtype}")
    target = np.dtype(dtype)
    if array.dtype != target:
        limit = np.iinfo(target)
        if array.size and (array.min() < limit.min or array.max() > limit.max):
            raise ValueError(f"{name} cannot be represented by {target} without overflow")
        array = array.astype(target, order="K", copy=False)
    return np.ascontiguousarray(array) if contiguous else array


def _bounded(array, minimum, maximum, name):
    if np.any(array < minimum) or np.any(array > maximum):
        raise ValueError(f"Packed {name} contains an out-of-range value")


def validate_rollout_inputs(states, beliefs, randoms, boundaries, initial_actions=None):
    """Return shape/dtype-checked arrays and exact branch boundaries.

    ``randoms`` is [games,625] in either C or Fortran order; preserving Fortran
    order lets its transpose transfer to CUDA without a large host copy. The
    public state validation here protects array/table bounds. Rule-level public
    state and first-action legality is also checked by the object-level callers.
    No malformed integer is clipped, wrapped or silently truncated.
    """
    shape = np.shape(states)
    if len(shape) != 2 or shape[1] != STATE_SIZE:
        raise ValueError(f"states must have shape [games,{STATE_SIZE}]")
    n = validate_batch_size(shape[0])
    states = _integers(states, (n, STATE_SIZE), np.int32, "states")
    randoms = _integers(randoms, (n, 625), np.uint32, "MT19937 streams", contiguous=False)
    beliefs = np.asarray(beliefs)
    if beliefs.shape != (n, 72) or beliefs.dtype.kind not in "fiu":
        raise ValueError("beliefs must be a numeric [games,72] array")
    beliefs = np.ascontiguousarray(beliefs, dtype=np.float64)
    if not np.isfinite(beliefs).all() or np.any(beliefs < 0):
        raise ValueError("beliefs must contain finite nonnegative probabilities")
    if n and (not np.allclose(beliefs[:, :24].sum(1), 1.0, rtol=0., atol=1e-10)
              or not np.allclose(beliefs[:, 24:].sum(1), 1.0, rtol=0., atol=1e-10)):
        raise ValueError("Each enhancement/coin posterior must have unit probability mass")
    if n and np.any(randoms[:, 624] > 624):
        raise ValueError("MT19937 position must be in [0,624]")
    if boundaries is None:
        # Complete catalogue evaluations do not group branch returns.
        boundaries = (0, n) if n else (0,)
    boundaries = tuple(require_integer(v, "branch boundary", maximum=n) for v in boundaries)
    if (not boundaries or boundaries[0] != 0 or boundaries[-1] != n or
            any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError("Branch boundaries must begin at zero, increase strictly and end at batch size")
    if n:
        # These fields index constant tables, packed rows or fixed-size features.
        limits = {
            "phase": (0, len(PHASES)-1), "node": (1, 46), "floor": (0, 20),
            "cap": (72, 144), "coins": (0, INT32_MAX), "paid": (0, 6),
            "restocks": (0, 2), "shop_return": (0, 2), "shop_ordinal": (0, 4),
            "h": (0, 10), "k": (0, 4), "seen": (0, (1 << 28)-1),
            "event_options": (0, (1 << 10)-1), "rarity": (0, 2), "mode": (0, 2),
            "bound": (0, 3), "source_return": (0, 2), "gE": (0, 5), "fE": (0, 3),
            "gC": (0, 5), "fC": (0, 3), "steps": (0, 1600), "error": (0, 0),
            "offer_n": (0, 3),
        }
        limits.update({k: (0, 1) for k in ("free", "first", "P", "Q", "A", "source", "reroll", "G1", "npc3_event")})
        columns = np.array([LAYOUT[key] for key in limits])
        low = np.array([pair[0] for pair in limits.values()], dtype=np.int32)
        high = np.array([pair[1] for pair in limits.values()], dtype=np.int32)
        fields = states[:, columns]
        if np.any(fields < low) or np.any(fields > high):
            for key, (minimum, maximum) in limits.items():
                _bounded(states[:, LAYOUT[key]], minimum, maximum, key)
        bound_source = (states[:, LAYOUT["source"]] == 1) & (states[:, LAYOUT["mode"]] == 2)
        if np.any(bound_source & (states[:, LAYOUT["bound"]] == 0)):
            raise ValueError("A bound-role source must select role 1, 2 or 3")
        potential_phase = np.isin(states[:, LAYOUT["phase"]],
                                 [PHASES.index(p) for p in ("DRAW", "PAGE", "ENHANCE_PAGE")])
        if np.any(potential_phase & (states[:, LAYOUT["source"]] == 0)):
            raise ValueError("A potential phase must retain its source before CUDA execution")
        events = states[:, LAYOUT["event"]]
        event_phase = states[:, LAYOUT["phase"]] == PHASES.index("EVENT")
        if np.any((events != 0) & ((events < 101) | (events > 128))) or np.any(event_phase & (events == 0)):
            raise ValueError("Packed event ID is outside the event table")
        pool_lengths = states[:, LAYOUT["pool_n"]:LAYOUT["pool_n"]+6]
        _bounded(pool_lengths, 0, np.array([3, 9, 3, 9, 3, 9]), "pool lengths")
        _bounded(states[:, LAYOUT["pool"]:LAYOUT["pool"]+54], 0, 6, "pool targets")
        inventory = states[:, LAYOUT["H"]:LAYOUT["H"]+252].reshape(n, 6, 42)
        _bounded(inventory, 0, 9, "inventory counts")
        if np.any(inventory.sum(2, dtype=np.int64) + pool_lengths != np.array([3, 9, 3, 9, 3, 9])):
            raise ValueError("Packed inventory/pools do not retain the complete 36-slot catalogue")
        stock = states[:, LAYOUT["stock"]:LAYOUT["stock"]+6]
        if np.any(stock < 0) or np.any(stock.sum(1, dtype=np.int64) > 8):
            raise ValueError("Packed stock exceeds the eight physical shelf slots")
        offer_n = states[:, LAYOUT["offer_n"]]
        selecting = np.isin(states[:, LAYOUT["phase"]], [PHASES.index("PAGE"), PHASES.index("ENHANCE_PAGE")])
        if np.any(selecting & (offer_n == 0)):
            raise ValueError("A selection phase must retain its visible candidates")
        offers = states[:, LAYOUT["offer"]:LAYOUT["offer"]+21].reshape(n, 3, 7)
        for slot in range(3):
            active = offer_n > slot
            if active.any():
                card = offers[active, slot]
                _bounded(card, np.array([0, 1, 1, 0, 0, 1, 0]),
                         np.array([1, 3, 2, 6, 5, 6, 9]), "visible offer")
                new = card[:, 0] == 1
                pool = 2*(card[:, 1]-1) + card[:, 2]-1
                lengths = pool_lengths[active, :][np.arange(len(card)), pool]
                if np.any(new & ((card[:, 6] < 1) | (card[:, 6] > lengths))):
                    raise ValueError("A visible new candidate refers to an invalid pool slot")
                if np.any(~new & ((card[:, 4] < 1) | (card[:, 6] != 0))):
                    raise ValueError("A visible old candidate has an invalid level/index")
    if initial_actions is not None:
        initial_actions = _integers(initial_actions, (n, 4), np.int32, "initial_actions")
        _bounded(initial_actions[:, 0], 0, len(KINDS)-1, "action kinds")
        for kind in np.unique(initial_actions[:, 0]):
            phases = states[initial_actions[:, 0] == kind, LAYOUT["phase"]]
            if not np.isin(phases, ACTION_PHASES[int(kind)]).all():
                raise ValueError("Initial action kind is incompatible with the public phase")
        if np.any(initial_actions[:, 1:] < 0):
            raise ValueError("Initial action parameters must be nonnegative")
        select = initial_actions[:, 0] == KINDS.index("SELECT")
        if np.any(select & ((initial_actions[:, 1] < 1) |
                            (initial_actions[:, 1] > states[:, LAYOUT["offer_n"]]))):
            raise ValueError("SELECT refers to an invalid visible candidate")
        buy = initial_actions[:, 0] == KINDS.index("BUY")
        if np.any(buy & ((initial_actions[:, 1] < 1) | (initial_actions[:, 1] > 3) |
                         (initial_actions[:, 3] < 1) | (initial_actions[:, 3] > 2))):
            raise ValueError("BUY refers to an invalid stock slot")
        event = initial_actions[:, 0] == KINDS.index("EVENT_OPTION")
        for eid in np.unique(states[event, LAYOUT["event"]]):
            mask = event & (states[:, LAYOUT["event"]] == eid)
            valid = [oid for oid in sim.EVENT_DATA[int(eid)][2] if oid]
            if not np.isin(initial_actions[mask, 1], valid).all():
                raise ValueError("EVENT_OPTION refers to an unknown event-table entry")
    return states, beliefs, randoms, boundaries, initial_actions
