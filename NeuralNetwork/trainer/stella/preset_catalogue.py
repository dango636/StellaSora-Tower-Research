"""The 4,960 equally weighted potential structures (stable zero-based IDs)."""
from __future__ import annotations
from functools import lru_cache
from itertools import product

CATALOGUE_SIZE = 4960
CATALOGUE_VERSION = 'structures4960_team84_132_v1'


def role_key(counts):
    return (sum(c*v for c,v in zip(counts,(1,3,6,1,3,6))), *counts)


def _role_options():
    out = []
    for rare in range(1,4):
        for selected in range(3,10):
            if rare == 1 and selected not in (3,4):
                continue
            if rare == 3 and selected not in (8,9):
                continue
            normal = selected-rare
            variants = [(0,0,rare,0,0,normal)]
            for low in (1,3):
                if rare:
                    variants.append((int(low==1),int(low==3),rare-1,0,0,normal))
                if normal:
                    variants.append((0,0,rare,int(low==1),int(low==3),normal-1))
            out.extend(c for c in variants if 18 <= role_key(c)[0] <= 54)
    return tuple(sorted(set(out), key=role_key))


@lru_cache(maxsize=1)
def catalogue_structures():
    roles = _role_options()
    result = []
    for main,a,b in product(roles, repeat=3):
        if role_key(a) < role_key(b):
            continue
        team = (main,a,b)
        if sum(c[0]+c[1]+c[3]+c[4] for c in team) > 1:
            continue
        if not 5 <= sum(sum(c[:3]) for c in team) <= 7:
            continue
        if 84 <= sum(role_key(c)[0] for c in team) <= 132:
            result.append(team)
    return tuple(sorted(result))


def structure_pools(structure):
    rows = []
    for counts in structure:
        for offset,size in ((0,3),(3,9)):
            one,three,six = counts[offset:offset+3]
            rows.append((6,)*six+(3,)*three+(1,)*one+(0,)*(size-one-three-six))
    return tuple(rows)


def _structure(pools):
    rows = tuple(tuple(row) for row in pools)
    if tuple(map(len,rows)) != (3,9,3,9,3,9):
        raise ValueError('Full original target vector must have pool lengths 3/9/3/9/3/9')
    if any(type(v) is not int or v not in (0,1,3,6) for row in rows for v in row):
        raise ValueError('Catalogue targets must be integers 0, 1, 3 or 6')
    roles = tuple(tuple(rows[2*r+p].count(v) for p in (0,1) for v in (1,3,6)) for r in range(3))
    return (roles[0], *sorted(roles[1:], key=role_key, reverse=True))


@lru_cache(maxsize=1)
def _index():
    return {structure:i for i,structure in enumerate(catalogue_structures())}


def catalogue_id(pools):
    """Validate full original targets and map pool/support permutations to one ID."""
    structure = _structure(pools)
    try:
        return _index()[structure]
    except KeyError:
        raise ValueError('Original target vector is outside the 4,960-structure catalogue') from None


def canonicalize_pools(pools):
    return structure_pools(catalogue_structures()[catalogue_id(pools)])


def validate_catalogue_pools(pools):
    """Validate without permuting the supplied vector."""
    rows = tuple(tuple(row) for row in pools)
    catalogue_id(rows)
    return rows


@lru_cache(maxsize=1)
def catalogue_presets():
    from .presets import Preset
    return tuple(Preset(f'C{i:04d}', structure_pools(s), 'catalogue4960')
                 for i,s in enumerate(catalogue_structures()))
