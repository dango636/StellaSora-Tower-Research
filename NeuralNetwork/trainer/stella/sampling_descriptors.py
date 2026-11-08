"""Device-side public hard-root categories, equivalent to hard_examples.

Only phase, offers, legal actions and the frozen teacher's chosen action are
read. No label returns, hidden latent values or terminal scores are involved.
Keeping categories on the device removes per-decision action/Q readbacks.
"""
from __future__ import annotations

import torch

from .features import PHASES, KINDS
from .hard_examples import SPECIFIC_KINDS


HARD_KIND_NAMES = (None, *SPECIFIC_KINDS, "shop_general", "page_general")


def hard_category_codes(phases, offers, offer_counts, actions, counts, choices):
    """Return category codes for a padded batch, on the input device.

    ``actions`` is [batch, MAX_ACTIONS, 4], and ``offers`` is [batch, 3, 7].
    Invalid padding is masked. Single-action/finished rows intentionally have
    no hard category, matching the original sampler's count > 1 condition.
    Ties are resolved upstream by the unchanged teacher action selection.
    """
    columns = torch.arange(actions.shape[1], device=actions.device)
    valid = columns[None, :] < counts[:, None]
    kinds, indices, prices = actions[:, :, 0], actions[:, :, 1], actions[:, :, 2]
    selected = actions.gather(1, choices.to(torch.int64)[:, None, None].expand(-1, 1, 4)).squeeze(1)
    chosen_kind, chosen_index, chosen_price = selected[:, 0], selected[:, 1], selected[:, 2]
    shop = phases == PHASES.index("SHOP")
    page = phases == PHASES.index("PAGE")
    page_general = page | (phases == PHASES.index("ENHANCE_PAGE"))
    result = torch.where(shop, 7, torch.where(page_general, 8, 0)).to(torch.int8)

    # Process ENHANCE first so BUY wins when RESTOCK/LEAVE can skip both.
    # This is the original classify_decision's BUY-before-ENHANCE order.
    leave_kind = KINDS.index("LEAVE")
    restock_kind = KINDS.index("RESTOCK")
    can_skip = (((kinds == leave_kind) | (kinds == restock_kind)) & valid).any(1)
    chosen_skip = (chosen_kind == leave_kind) | (chosen_kind == restock_kind)
    for kind, price, selected_code, skip_code in (("ENHANCE", 180, 5, 6), ("BUY", 160, 3, 4)):
        kind_id = KINDS.index(kind)
        available = ((kinds == kind_id) & (prices == price) & valid).any(1)
        relevant = shop & can_skip & available
        result = torch.where(relevant & chosen_skip, skip_code, result)
        result = torch.where(relevant & (chosen_kind == kind_id) & (chosen_price == price), selected_code, result)

    offer_columns = torch.arange(offers.shape[1], device=offers.device)
    good = ((offer_columns[None, :] < offer_counts[:, None]) & (offers[:, :, 0] == 1)
            & (offers[:, :, 3] == 3) & (offers[:, :, 5] == 3))
    select_good = (((indices[:, :, None] == (offer_columns+1)[None, None, :])
                    & good[:, None, :]).any(2) & (kinds == KINDS.index("SELECT")) & valid)
    chosen_good = ((chosen_index[:, None] == (offer_columns+1)[None, :]) & good).any(1)
    reroll = KINDS.index("REROLL")
    relevant = page & ((kinds == reroll) & valid).any(1) & select_good.any(1)
    result = torch.where(relevant & (chosen_kind == reroll), 2, result)
    result = torch.where(relevant & (chosen_kind == KINDS.index("SELECT")) & chosen_good, 1, result)
    return torch.where(counts > 1, result, 0).to(torch.int8)
