from dataclasses import replace
from fractions import Fraction as F
from random import Random
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.vendor import stella_sim as sim
from stella.gpu_game_tables import pack_state, unpack_state
from stella.features import encode_state, _slices


def test_npc3_event_callback_and_state_roundtrip():
    def chance(flag):
        return sum(p for p, d, _ in sim._coin_paths(7, 2, False, 0, 0, 1, flag)
                   if d[4] == "EVENT_DRAW")
    assert chance(False) == F(80, 101)
    assert chance(True) == F(50, 101)
    state = replace(sim.initial_843(), npc3EventReward=True)
    assert unpack_state(*pack_state(state)).npc3EventReward


def test_event_reward_marks_only_early_potential_creation():
    state = replace(sim.initial_843(), node=5, eventId=101, seen=(101,))
    reward = sim._event_option(state, sim.Action("EVENT_OPTION", 10101), Random(1))
    assert reward.npc3EventReward
    assert not sim._event_option(replace(state, node=8), sim.Action("EVENT_OPTION", 10101), Random(1)).npc3EventReward


def test_feature_folding_preserves_actual_posterior():
    belief = ((0, 0, 0, F(1, 4)), (0, 0, 1, F(3, 4)))
    state = replace(sim.initial_843(), cBel=belief, npc3EventReward=True)
    coin = encode_state(state)[_slices["coin_posterior"]]
    assert np.array_equal(coin[:2], [1, 0])
    assert state.cBel == belief


def test_first_shop_restock_uses_first_shop_b():
    base = replace(sim.initial_843(), shopOrdinal=1)
    later = replace(base, shopOrdinal=2)
    first = sim._shelf(base, Random(5), False)
    other = sim._shelf(later, Random(5), False)
    assert sum(first.stock[1]) == sum(other.stock[1]) + 1
    assert sum(first.stock[0]) == sum(other.stock[0]) - 1
