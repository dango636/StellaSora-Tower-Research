"""Public decision categories; no rollout results enter root selection."""
from __future__ import annotations

SPECIFIC_KINDS = ("lv3_take", "lv3_reroll", "shop_buy160", "shop_buy160_skip",
                  "shop_enhance180", "shop_enhance180_skip")

def classify_decision(phase, offers, actions, chosen_index):
    """Actions are (kind, offer index, price); chosen index is teacher argmax."""
    chosen = actions[chosen_index][0]
    kinds = {a[0] for a in actions}
    if phase == "PAGE" and "REROLL" in kinds:
        good = {i+1 for i, c in enumerate(offers) if c[0] == 1 and c[3] == 3 and c[5] == 3}
        if any(a[0] == "SELECT" and a[1] in good for a in actions):
            if chosen == "REROLL":
                return "lv3_reroll"
            if chosen == "SELECT" and actions[chosen_index][1] in good:
                return "lv3_take"
    if phase == "SHOP" and kinds.intersection(("RESTOCK", "LEAVE")):
        for kind, price, name in (("BUY", 160, "shop_buy160"), ("ENHANCE", 180, "shop_enhance180")):
            if any(a[0] == kind and a[2] == price for a in actions):
                selected = actions[chosen_index]
                if selected[0] == kind and selected[2] == price:
                    return name
                if chosen in ("RESTOCK", "LEAVE"):
                    return name + "_skip"
    return "shop_general" if phase == "SHOP" else ("page_general" if phase in ("PAGE", "ENHANCE_PAGE") else None)

def classify_hard_state(state, actions, chosen_index):
    """Classify old label states using the teacher's actual chosen action."""
    return classify_decision(state.phase, state.offer,
                             [(a.kind, a.index, a.price) for a in actions], chosen_index)

def choose_hard_root(task, remaining, gap, classify):
    from random import Random
    from .presets import canonical_seed
    groups = {}
    for item in remaining:
        category = classify(item[1])
        if category:
            groups.setdefault(category, []).append(item)
    specific = [name for name in SPECIFIC_KINDS if name in groups]
    categories = specific or sorted(groups)
    if not categories:
        return min(remaining[:6], key=lambda item: gap(item[1]))
    index = task.get("phase_sampling_index", task["ordinal"])
    # Development uses equal public category quotas, never mined train weights.
    weights = task.get("hard_category_weights", {}) if task.get("split") != "val" else {}
    if weights:
        rng = Random(canonical_seed("hard_category", task["selection_seed"]))
        category = rng.choices(categories, weights=[max(.01, float(weights.get(k, 1))) for k in categories])[0]
    else:
        category = categories[index % len(categories)]
    band = (index // max(1, len(categories))) % 3
    def rank(item):
        state = item[1]
        actual = 0 if state.floor <= 7 else (1 if state.floor <= 14 else 2)
        return actual != band, gap(state)
    return min(groups[category], key=rank)
