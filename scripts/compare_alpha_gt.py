#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

# Reuse existing helpers from the engine
from quant_engine.run_day import _load_inputs          # loads prices, fundamentals, sectors, prev_w, asof
from quant_engine.data_io import unique_dates
from quant_engine.signals import momentum_12m_1m_gap, value_ep
from quant_engine.prep import winsorize, zscore, sector_neutralize

def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_alpha_gt_logged(path: Path, asof: pd.Timestamp) -> pd.Series:
    """
    Expect CSV with columns: asof_dt, ticker, alpha_gt_used (+ optional mom_z_tm1, val_z_tm1, betas).
    Returns z-scored Series indexed by ticker for 'asof'. Empty Series if not found.
    """
    if not Path(path).exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    if "asof_dt" not in df or "ticker" not in df or "alpha_gt_used" not in df:
        raise ValueError("alpha_gt CSV must have columns: asof_dt, ticker, alpha_gt_used")
    df["asof_dt"] = pd.to_datetime(df["asof_dt"]).dt.normalize()
    s = df.loc[df["asof_dt"] == asof, ["ticker","alpha_gt_used"]].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    ser = s.set_index("ticker")["alpha_gt_used"].astype(float)
    # z-score so scale doesn't matter in corr/RMSE/ratio
    sd = ser.std(ddof=1)
    if sd and np.isfinite(sd) and sd > 0:
        ser = (ser - ser.mean()) / sd
    else:
        ser = ser * 0.0
    return ser

def build_engine_combined_signal(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    sectors_map: pd.Series | dict | None,
    asof: pd.Timestamp,
    *,
    mom_lookback: int,
    mom_gap: int,
    val_min_lag_days: int,
    prep_cfg: dict | None,
    w_mom: float,
    w_val: float,
) -> pd.Series:
    """
    Rebuild the engine's combined alpha at a given as-of date using the same pipeline
    as run_day: momentum(12-1 with gap), value E/P (PIT), winsorize->zscore->(optional) sector-neutralize, combine -> zscore.
    """
    mom = momentum_12m_1m_gap(prices_df, asof_dt=asof, lookback=mom_lookback, gap=mom_gap)
    val = value_ep(fundamentals_df, prices_df, asof_dt=asof, min_lag_days=val_min_lag_days)
    idx = mom.index.intersection(val.index)
    if idx.empty:
        return pd.Series(dtype=float)
    mom = mom.reindex(idx)
    val = val.reindex(idx)

    # Preprocessing knobs
    p_low, p_high = 0.01, 0.99
    sector_neut = True
    if prep_cfg:
        if "winsor_pcts" in prep_cfg and isinstance(prep_cfg["winsor_pcts"], (list, tuple)) and len(prep_cfg["winsor_pcts"]) == 2:
            p_low, p_high = float(prep_cfg["winsor_pcts"][0]), float(prep_cfg["winsor_pcts"][1])
        elif "winsorize_percentile" in prep_cfg:
            p = float(prep_cfg["winsorize_percentile"]); p_low, p_high = p, 1.0 - p
        sector_neut = bool(prep_cfg.get("sector_neutralize", True))

    mom_p = zscore(winsorize(mom, p_low, p_high))
    val_p = zscore(winsorize(val, p_low, p_high))

    if sector_neut and sectors_map is not None:
        sec = pd.Series(sectors_map).reindex(idx)
        mom_p = sector_neutralize(mom_p, sec)
        val_p = sector_neutralize(val_p, sec)

    comb = w_mom * mom_p + w_val * val_p
    return zscore(comb).dropna()

def align_series(engine_sig: pd.Series, gt_sig: pd.Series) -> dict:
    """
    Align by tickers and compute: Spearman corr, RMSE, scale ratio (std(engine)/std(gt)), and n.
    """
    idx = engine_sig.index.intersection(gt_sig.index)
    if idx.size < 3:
        return {"corr": np.nan, "rmse": np.nan, "scale_ratio": np.nan, "n": int(idx.size)}
    e = engine_sig.reindex(idx)
    g = gt_sig.reindex(idx)
    corr = float(pd.Series(e).corr(pd.Series(g), method="spearman"))
    rmse = float(np.sqrt(np.mean((e.values - g.values)**2)))
    se = float(np.std(e.values, ddof=1)); sg = float(np.std(g.values, ddof=1))
    ratio = float(se/sg) if sg > 0 else np.nan
    return {"corr": corr, "rmse": rmse, "scale_ratio": ratio, "n": int(idx.size)}

def summarize(df: pd.DataFrame) -> dict:
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["corr"])
    if d.empty:
        return {"mean_corr": np.nan, "median_corr": np.nan, "p10_corr": np.nan, "mean_scale_ratio": np.nan, "n_days": 0}
    return {
        "mean_corr": float(d["corr"].mean()),
        "median_corr": float(d["corr"].median()),
        "p10_corr": float(d["corr"].quantile(0.10)),
        "mean_scale_ratio": float(d["scale_ratio"].dropna().mean()) if "scale_ratio" in d else np.nan,
        "n_days": int(d.shape[0]),
    }

def main():
    ap = argparse.ArgumentParser(description="Compare engine combined alpha to ground-truth alpha and produce a report.")
    ap.add_argument("--config", required=True, help="Path to engine config YAML")
    ap.add_argument("--alpha-gt-path", default="data/alpha_gt.csv", help="Path to ground-truth alpha CSV")
    ap.add_argument("--asof", help="Anchor as-of (YYYY-MM-DD). If omitted, use latest with t-1 & t+1.")
    ap.add_argument("--window", type=int, default=60, help="Days to include ending at anchor (default 60)")
    ap.add_argument("--outdir", default="reports", help="Directory for outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Load inputs exactly like the engine
    prices_df, fundamentals_df, sectors_ser, prev_w, asof0 = _load_inputs(cfg, args.asof or "")
    all_dates = unique_dates(prices_df)
    if args.asof:
        anchor = pd.to_datetime(args.asof).normalize()
        if anchor not in set(all_dates):
            print(f"ERROR: asof {anchor.date()} not found in prices.", file=sys.stderr)
            sys.exit(1)
    else:
        anchor = all_dates.max()

    # Build a window of dates t where t-1 and t+1 exist
    all_set = set(all_dates)
    dates = []
    for t in reversed(all_dates[all_dates <= anchor]):
        if (t - pd.tseries.offsets.BDay(1)) in all_set and (t + pd.tseries.offsets.BDay(1)) in all_set:
            dates.append(t)
        if len(dates) >= args.window:
            break
    dates = list(reversed(dates))
    if not dates:
        print("No eligible dates with both t-1 and t+1.", file=sys.stderr)
        sys.exit(1)

    # Engine knobs
    sig_cfg = cfg.get("signals", {})
    mom_lb = int(sig_cfg.get("momentum", {}).get("lookback", sig_cfg.get("momentum", {}).get("lookback_days", 252)))
    mom_gap = int(sig_cfg.get("momentum", {}).get("gap", sig_cfg.get("momentum", {}).get("gap_days", 21)))
    val_lag = int(sig_cfg.get("value", {}).get("min_lag_days", 60))
    w_mom = float(sig_cfg.get("weights", {}).get("momentum", 0.5))
    w_val = float(sig_cfg.get("weights", {}).get("value", 0.5))
    prep_cfg = cfg.get("prep", cfg.get("signals_legacy", {}))

    # Iterate dates and compute alignment
    rows = []
    for t in dates:
        tm1 = t - pd.tseries.offsets.BDay(1)
        eng = build_engine_combined_signal(
            prices_df, fundamentals_df, sectors_ser, tm1,
            mom_lookback=mom_lb, mom_gap=mom_gap, val_min_lag_days=val_lag,
            prep_cfg=prep_cfg, w_mom=w_mom, w_val=w_val
        )
        gt = load_alpha_gt_logged(Path(args.alpha_gt_path), t)
        if gt.empty:
            # If GT missing for this date, skip row (keeps report honest).
            continue
        stats = align_series(eng, gt)
        rows.append({"asof_dt": t, **stats})

    if not rows:
        print("No overlap between engine signals and ground-truth alpha in the window.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("asof_dt")
    summ = summarize(df)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"alpha_gt_alignment_{dates[-1].date().isoformat()}.csv"
    txt_path = outdir / f"alpha_gt_alignment_{dates[-1].date().isoformat()}.txt"
    df.to_csv(csv_path, index=False)

    with open(txt_path, "w") as f:
        f.write("ALPHA vs GROUND-TRUTH ALIGNMENT REPORT\n")
        f.write(f"Anchor: {dates[-1].date().isoformat()}  Window: {len(df)} days\n")
        f.write("="*60 + "\n")
        f.write(f"Summary: mean_corr={summ['mean_corr']:.3f}, median_corr={summ['median_corr']:.3f}, "
                f"p10_corr={summ['p10_corr']:.3f}, mean_scale={summ['mean_scale_ratio']:.3f}, "
                f"n_days={summ['n_days']}\n\n")
        f.write("First/last 3 rows:\n")
        f.write(df.head(3).to_string(index=False) + "\n...\n" + df.tail(3).to_string(index=False) + "\n")

    print(f"Wrote {txt_path} and {csv_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 