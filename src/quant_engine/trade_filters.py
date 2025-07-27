from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd

EPS = 1e-12

def small_trade_mask(
    delta_w: pd.Series,
    prices: Optional[pd.Series] = None,
    aum: Optional[float] = None,
    *,
    min_weight: float = 0.0005,     # 5 bps default
    min_notional: Optional[float] = None
) -> pd.Series:
    """
    Return a boolean mask indexed by ticker where True = 'freeze (too small to trade)'.
    Logic:
      - Always freeze if abs(delta_w) < min_weight.
      - If min_notional is provided AND prices & aum are available, also freeze when
        abs(delta_w) * aum < min_notional.
    Notes:
      - prices should be aligned by ticker as of t (or t-1); only its level matters.
      - Any missing price or non-finite delta -> treat as small (freeze).
    """
    dw = delta_w.astype(float).copy()
    mask = dw.abs() < float(min_weight)
    if min_notional is not None and prices is not None and aum is not None:
        pr = prices.reindex(dw.index).astype(float)
        notional = dw.abs() * float(aum) * pr / pr.clip(lower=EPS)  # ~ aum * abs(dw) (price cancels if normalized elsewhere)
        # If price is missing or zero, treat as small
        extra = (~np.isfinite(pr)) | (~np.isfinite(dw)) | (notional < float(min_notional))
        mask = mask | extra
    else:
        # Missing inputs -> be conservative for NaNs
        mask = mask | (~np.isfinite(dw))
    return mask.fillna(True)

def apply_no_trade_band(
    prev_w: pd.Series,
    new_w: pd.Series,
    prices: Optional[pd.Series] = None,
    aum: Optional[float] = None,
    *,
    min_weight: float = 0.0005,
    min_notional: Optional[float] = None
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """
    Compute delta, build mask of small trades, and return:
      new_w_frozen, freeze_mask, stats
    Where new_w_frozen = new_w with small trades 'frozen' back to prev_w (no change).
    stats: {"n_frozen": int, "turnover_before": float, "turnover_after": float}
    """
    # Ensure aligned indices
    idx = prev_w.index.union(new_w.index)
    pw = prev_w.reindex(idx).fillna(0.0)
    nw = new_w.reindex(idx).fillna(0.0)

    delta = nw - pw
    mask = small_trade_mask(delta, prices=prices.reindex(idx) if prices is not None else None,
                            aum=aum, min_weight=min_weight, min_notional=min_notional)

    turnover_before = 0.5 * float(np.abs(delta).sum())
    nw_frozen = nw.where(~mask, pw)  # replace small-trade names with previous weights
    turnover_after = 0.5 * float(np.abs(nw_frozen - pw).sum())

    stats = {
        "n_frozen": int(mask.sum()),
        "turnover_before": float(turnover_before),
        "turnover_after": float(turnover_after),
    }
    return nw_frozen, mask.astype(bool), stats 