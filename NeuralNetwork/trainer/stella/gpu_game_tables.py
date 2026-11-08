"""Public frozen-rule tables shared by the CUDA simulator and Python adapter."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import os
import tempfile
from pathlib import Path

import numpy as np

from . import features as ft
from .vendor import stella_sim as sim

MAX_ACTIONS = 5
FIELDS = ("phase node floor cap coins free paid restocks first shop_return shop_ordinal P Q h A k seen "
          "event event_options source rarity mode bound source_return reroll gE fE gC fC G1 npc3_event steps root error").split()
LAYOUT = {name: i for i, name in enumerate(FIELDS)}
for name, size in (("stock", 6), ("pool_n", 6), ("pool", 54), ("H", 252), ("offer_n", 1), ("offer", 21)):
    LAYOUT[name] = len(FIELDS)
    FIELDS.extend([name]*size)
STATE_SIZE = len(FIELDS)


def pack_state(s, root=0):
    a = np.zeros(STATE_SIZE, dtype=np.int32)
    values = dict(phase=ft.PHASES.index(s.phase), node=s.node, floor=s.floor, cap=s.cap, coins=s.coins,
                  free=s.freeEnhance, paid=s.paidIndex, restocks=s.restocks, first=s.firstInitial,
                  shop_return=ft.RETURNS.index(s.shopReturnPhase), shop_ordinal=s.shopOrdinal,
                  npc3_event=s.npc3EventReward, P=s.P, Q=s.Q, h=s.h, A=s.A, k=s.k, seen=sum(1 << (eid-101) for eid in s.seen),
                  event=s.eventId, event_options=sum(1 << (oid % 100) for oid in s.eventOptions), root=root,
                  source=s.source is not None, offer_n=len(s.offer))
    if s.source:
        values.update(rarity=ft.RARITIES.index(s.source.rarity), mode=ft.MODES.index(s.source.roleMode),
                      bound=s.source.boundRole, source_return=ft.RETURNS.index(s.source.returnPhase),
                      reroll=s.source.rerollAllowed)
    for name, value in values.items():
        a[LAYOUT[name]] = value
    a[LAYOUT["stock"]:LAYOUT["stock"]+6] = np.asarray(s.stock).ravel()
    for i, pool in enumerate(s.pools):
        a[LAYOUT["pool_n"]+i] = len(pool)
        a[LAYOUT["pool"]+i*9:LAYOUT["pool"]+i*9+len(pool)] = pool
    for r, q, t, level, n in s.H:
        a[LAYOUT["H"]+((r-1)*2+q-1)*42+t*6+level-1] = n
    a[LAYOUT["offer"]:LAYOUT["offer"]+len(s.offer)*7] = np.asarray(s.offer).reshape(-1)
    belief = np.zeros(72, dtype=np.float64)
    for g, f, p in s.eBel:
        belief[g*4+f] = float(p)
    for g, f, first, p in s.cBel:
        belief[24+(g*4+f)*2+first] = float(p)
    return a, belief


def unpack_state(packed, belief):
    """Restore only public fields, normalizing the FP64 posterior exactly."""
    a = [int(value) for value in packed]
    def get(name):
        return a[LAYOUT[name]]
    def posterior(values, coin=False):
        mass = [(i, Fraction(float(p))) for i, p in enumerate(values) if p > 0]
        total = sum((p for _, p in mass), Fraction())
        return tuple(((i//8, i//2 % 4, i % 2) if coin else (i//4, i % 4)) + (p/total,)
                     for i, p in mass)
    source = (sim.Source(ft.RARITIES[get("rarity")], ft.MODES[get("mode")], get("bound"),
                         ft.RETURNS[get("source_return")], bool(get("reroll")))
              if get("source") else None)
    inventory = tuple((i//84+1, i//42 % 2+1, i//6 % 7, i % 6+1, count)
                      for i, count in enumerate(a[LAYOUT["H"]:LAYOUT["H"]+252]) if count)
    pools = tuple(tuple(a[LAYOUT["pool"]+9*i:LAYOUT["pool"]+9*i+a[LAYOUT["pool_n"]+i]])
                  for i in range(6))
    offers = tuple(tuple(a[LAYOUT["offer"]+7*i:LAYOUT["offer"]+7*i+7])
                   for i in range(get("offer_n")))
    event = get("event")
    options = (tuple(oid for oid in sim.EVENT_DATA[event][1]
                     if get("event_options") & (1 << (oid % 100))) if event else ())
    return sim.PublicState(
        phase=ft.PHASES[get("phase")], node=get("node"), floor=get("floor"),
        H=inventory, pools=pools, cap=get("cap"), P=bool(get("P")), Q=bool(get("Q")), h=get("h"),
        npc3EventReward=bool(get("npc3_event")), eBel=posterior(belief[:24]), cBel=posterior(belief[24:], coin=True),
        coins=get("coins"), freeEnhance=bool(get("free")), paidIndex=get("paid"),
        stock=tuple(tuple(a[LAYOUT["stock"]+3*i:LAYOUT["stock"]+3*i+3]) for i in range(2)),
        restocks=get("restocks"), firstInitial=bool(get("first")),
        shopReturnPhase=ft.RETURNS[get("shop_return")], shopOrdinal=get("shop_ordinal"),
        offer=offers, source=source, A=bool(get("A")), k=get("k"),
        seen=tuple(eid for eid in ft.EVENTS if get("seen") & (1 << (eid-101))),
        eventId=event, eventOptions=options,
    )


def action_array(a):
    return (ft.KINDS.index(a.kind), a.index, a.price, a.type)


@lru_cache(maxsize=1)
def coin_tables():
    """Enumerate the original exact categorical draws, keeping their order."""
    directory = Path(os.environ.get("STELLA_GPU_CACHE_DIR") or
                     Path(__file__).resolve().parents[1] / "runtime/cuda-cache")
    cache = directory / f"coin-{sim.MODEL_VERSION}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return tuple(data[k] for k in ("index", "paths", "probabilities", "descriptors"))
    index = np.zeros((47, 5, 2, 2, 48, 3), dtype=np.int32)
    paths, probabilities, descriptors, identities = [], [], [], {}
    for node in range(1, 47):
        for k in range(5):
            for A in (False, True):
                for event_reward in (False, True):
                    for latent in range(48):
                        g, f, first = latent//8, latent//2 % 4, latent % 2
                        outcomes = sim._coin_paths(node, k, A, g, f, first, event_reward)
                        weights = sim._fraction_weights(tuple(p for p, _, _ in outcomes))
                        index[node, k, int(A), int(event_reward), latent] = len(paths), len(outcomes), sum(weights)
                        for (p, desc, nxt), weight in zip(outcomes, weights):
                            if desc not in identities:
                                identities[desc] = len(descriptors)
                                n, floor, kk, aa, phase, delta, shop = desc
                                descriptors.append((n, floor, kk, int(aa), ft.PHASES.index(phase), delta, shop))
                            paths.append((weight, identities[desc], (nxt[0]*4+nxt[1])*2+nxt[2]))
                            probabilities.append(float(p))
    arrays = (index, np.asarray(paths, np.int32), np.asarray(probabilities, np.float64),
              np.asarray(descriptors, np.int32))
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache.parent, prefix="coin-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **dict(zip(("index", "paths", "probabilities", "descriptors"), arrays)))
        os.replace(temporary, cache)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return arrays


def header():
    out = [f"#define S_{k} {v}" for k, v in LAYOUT.items()]
    out += [f"#define F_{b['name']} {b['start']}" for b in ft._blocks]
    out += [f"#define PH_{name} {i}" for i, name in enumerate(ft.PHASES)]
    out += [f"#define AC_{name} {i}" for i, name in enumerate(ft.KINDS)]
    out += [f"#define STATE_SIZE {STATE_SIZE}", f"#define STATE_DIM {ft.STATE_DIM}",
            f"#define FEATURE_DIM {ft.FEATURE_DIM}", f"#define MAX_ACTIONS {MAX_ACTIONS}"]
    def array(name, values, kind="int"):
        value = np.asarray(values)
        shape = "".join(f"[{n}]" for n in value.shape)
        def literal(x):
            if isinstance(x, list):
                return "{" + ",".join(map(literal, x)) + "}"
            return repr(float(x)) + "f" if kind == "float" else repr(float(x)) if kind == "double" else str(int(x))
        out.append(f"__device__ __constant__ {kind} {name}{shape} = {literal(value.tolist())};")
    array("PRICES", sim.PRICES)
    array("DRINK_PRICES", sim.DRINK_PRICES)
    array("THRESHOLDS", ft.THRESHOLDS)
    array("POOL_LENGTHS", sim.INITIAL_POOL_LENGTHS)
    array("EVENT_WEIGHTS", [sim.EVENT_DATA[e][0] for e in ft.EVENTS])
    array("OPTION_IDS", ft.OPTIONS)
    options = np.zeros((29, 10, 7), dtype=np.int32)
    expectations = np.zeros((29, 10, 10), dtype=np.float32)
    utilities = np.zeros((29, 10), dtype=np.float64)
    outcomes = []
    masks = []
    dominance = np.zeros((29, 10), np.int32)
    for eid in ft.EVENTS:
        _, predictive, entries = sim.EVENT_DATA[eid]
        masks.append(sum(1 << (oid % 100) for oid in predictive))
        for better, worse in sim.DOMINANCE.get(eid, ()):
            dominance[eid-100, worse % 100] |= 1 << (better % 100)
        for oid, (cost, results) in entries.items():
            slot = oid % 100
            weights = sim._fraction_weights(tuple(Fraction(r[0], r[1]) for r in results))
            options[eid-100, slot] = cost, len(outcomes), len(results), sum(weights), min(r[2] for r in results), ft.OPTIONS.index(oid), oid
            desc = expectations[eid-100, slot]
            utilities[eid-100, slot] = sum(n/d * (delta/100 + int(rarity != "none"))
                                          for n, d, delta, rarity, _ in results)
            for (n, d, delta, rarity, scope), weight in zip(results, weights):
                outcomes.append((weight, delta, ("none", *ft.RARITIES).index(rarity),
                                 ("none", "party", "main", "support_random").index(scope)))
                probability = n/d
                desc[0] += probability*delta/500
                desc[1] += probability*abs(delta)/500
                desc[2] += probability*(delta/500)**2
                desc[3+("none", *ft.RARITIES).index(rarity)] += probability
                if scope != "none":
                    desc[7+("party", "main", "support_random").index(scope)] += probability
    array("EVENT_MASKS", masks)
    array("EVENT_OPTIONS", options)
    array("EVENT_OUTCOMES", outcomes)
    array("EVENT_EXPECTATIONS", expectations, "float")
    array("EVENT_UTILITIES", utilities, "double")
    array("EVENT_DOMINANCE", dominance)
    return "\n".join(out) + "\n"
