#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from pandas.tseries.offsets import BDay

# Reuse your engine helpers so the feature construction matches the engine exactly
from quant_engine.run_day import _load_inputs            # loads prices, fundamentals, sectors, prev_w, asof
from quant_engine.data_io import unique_dates
from quant_engine.signals import momentum_12m_1m_gap, value_ep
from quant_engine.prep import winsorize, zscore, sector_neutralize

def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_alpha_gt_for_date(alpha_gt_path: Path, t: pd.Timestamp) -> pd.DataFrame:
    """
    Load logged ground-truth alpha for date t.
    Expects columns: asof_dt, ticker, alpha_gt_used (and optionally mom_z_tm1, val_z_tm1).
    Returns DataFrame index=ticker with columns: alpha_gt_used[, mom_z_tm1, val_z_tm1]
    """
    if not alpha_gt_path.exists():
        raise FileNotFoundError(f"alpha_gt file not found: {alpha_gt_path}")
    df = pd.read_csv(alpha_gt_path)
    if not {"asof_dt","ticker","alpha_gt_used"}.issubset(df.columns):
        raise ValueError("alpha_gt CSV must include columns: asof_dt, ticker, alpha_gt_used")
    df["asof_dt"] = pd.to_datetime(df["asof_dt"]).dt.normalize()
    sub = df.loc[df["asof_dt"] == t, ["ticker","alpha_gt_used"] + [c for c in ["mom_z_tm1","val_z_tm1"] if c in df.columns]]
    if sub.empty:
        return pd.DataFrame(index=pd.Index([], name="ticker"))
    sub = sub.set_index("ticker").astype(float)
    # z-score alpha cross-sectionally for stable regression (scale-free)
    a = sub["alpha_gt_used"]
    sd = a.std(ddof=1)
    if np.isfinite(sd) and sd > 0:
        sub["alpha_z"] = (a - a.mean()) / sd
    else:
        sub["alpha_z"] = 0.0
    return sub

def build_engine_features(prices_df: pd.DataFrame,
                          fundamentals_df: pd.DataFrame,
                          sectors_ser: pd.Series | None,
                          t: pd.Timestamp,
                          *,
                          mom_lb: int, mom_gap: int, val_min_lag_days: int,
                          winsor_pcts: tuple[float,float] | None,
                          sector_mode: str) -> pd.DataFrame:
    """
    Build engine features (at t-1):
      columns: mom_z, val_z (cross-sectional z-scores)
    sector_mode: 'none' | 'neutralize' | 'fixedeffects'
      - 'neutralize': apply sector_neutralize to each column (demean by sector using regression residuals)
      - 'fixedeffects': subtract sector means from BOTH features and alpha later (handled in projection)
    """
    tm1 = t - BDay(1)
    mom = momentum_12m_1m_gap(prices_df, asof_dt=tm1, lookback=mom_lb, gap=mom_gap)
    val = value_ep(fundamentals_df, prices_df, asof_dt=tm1, min_lag_days=val_min_lag_days)

    idx = mom.index.intersection(val.index)
    if idx.empty:
        return pd.DataFrame(index=pd.Index([], name="ticker"), columns=["mom_z","val_z"])

    mom = mom.reindex(idx)
    val = val.reindex(idx)

    if winsor_pcts is not None:
        lo, hi = winsor_pcts
        mom = winsorize(mom, lo, hi)
        val = winsorize(val, lo, hi)

    mom_z = zscore(mom)
    val_z = zscore(val)

    if sector_mode == "neutralize" and sectors_ser is not None:
        sec = sectors_ser.reindex(idx)
        mom_z = sector_neutralize(mom_z, sec)
        val_z = sector_neutralize(val_z, sec)

    feats = pd.DataFrame({"mom_z": mom_z, "val_z": val_z})
    return feats.dropna()

def sector_demean(df: pd.DataFrame, sectors: pd.Series, cols: list[str]) -> pd.DataFrame:
    """Simple 'fixed effects': subtract sector means from selected columns."""
    out = df.copy()
    sec = sectors.reindex(out.index)
    for c in cols:
        out[c] = out[c] - out.groupby(sec)[c].transform("mean")
    return out

def ols_capture(X: pd.DataFrame, y: pd.Series, intercept: bool = True) -> dict:
    """
    Solve cross-sectional OLS y ~ [1] + X.
    Returns dict with: r2, n, betas={col: coef}, intercept.
    """
    # Align
    idx = X.index.intersection(y.index)
    X = X.reindex(idx).astype(float)
    y = y.reindex(idx).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan).dropna()
    y = y.loc[X.index]
    n = X.shape[0]
    if n < max(5, X.shape[1] + 2):
        return {"r2": np.nan, "n": int(n), "betas": {c: np.nan for c in X.columns}, "intercept": np.nan}
    # Design
    if intercept:
        A = np.column_stack([np.ones(n), X.values])
    else:
        A = X.values
    # Solve via least-squares
    coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
    if intercept:
        c0, b = float(coef[0]), coef[1:]
    else:
        c0, b = 0.0, coef
    yhat = (A @ coef)
    # R^2
    ybar = float(y.mean())
    ss_tot = float(np.sum((y.values - ybar)**2))
    ss_res = float(np.sum((y.values - yhat)**2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res/ss_tot
    betas = {col: float(val) for col, val in zip(X.columns, b)}
    return {"r2": float(r2), "n": int(n), "betas": betas, "intercept": float(c0)}

def main():
    ap = argparse.ArgumentParser(description="Daily cross-sectional alpha capture (projection) onto chosen signals.")
    ap.add_argument("--config", required=True, help="Path to engine config YAML")
    ap.add_argument("--alpha-gt-path", default="data/alpha_gt.csv", help="Path to ground-truth alpha CSV")
    ap.add_argument("--asof", help="Anchor as-of (YYYY-MM-DD). If omitted, use last available.")
    ap.add_argument("--window", type=int, default=60, help="Number of business days ending at anchor (default 60)")
    ap.add_argument("--outdir", default="reports", help="Directory for outputs")
    # Feature construction knobs
    ap.add_argument("--features-source", choices=["engine","logged"], default="engine",
                    help="Use engine-built features at t-1, or logged mom_z_tm1/val_z_tm1 if available.")
    ap.add_argument("--winsor-pcts", type=float, nargs=2, metavar=("LOW","HIGH"),
                    default=None, help="Winsorize raw features before z-scoring (e.g., 0.01 0.99).")
    ap.add_argument("--sector-mode", choices=["none","neutralize","fixedeffects"], default="none",
                    help="How to handle sector structure in features/alpha for projection.")
    ap.add_argument("--intercept", action="store_true", help="Include an intercept in the daily regression.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Load inputs the same way as the engine
    prices_df, fundamentals_df, sectors_ser, prev_w, asof0 = _load_inputs(cfg, args.asof or "")
    all_dates = unique_dates(prices_df)
    if args.asof:
        anchor = pd.to_datetime(args.asof).normalize()
        if anchor not in set(all_dates):
            print(f"ERROR: asof {anchor.date()} not in price calendar.", file=sys.stderr)
            return 1
    else:
        anchor = all_dates.max()

    # Build window with t-1 and t+1 available (optional but consistent with other diagnostics)
    all_set = set(all_dates)
    dates = []
    for t in reversed(all_dates[all_dates <= anchor]):
        if (t - BDay(1)) in all_set and (t + BDay(1)) in all_set:
            dates.append(t)
        if len(dates) >= args.window:
            break
    dates = list(reversed(dates))
    if not dates:
        print("No eligible dates with both t-1 and t+1.", file=sys.stderr)
        return 1

    # Pull engine knobs
    sig_cfg = cfg.get("signals", {})
    mom_lb = int(sig_cfg.get("momentum", {}).get("lookback", 252))
    mom_gap = int(sig_cfg.get("momentum", {}).get("gap", 21))
    val_lag = int(sig_cfg.get("value", {}).get("min_lag_days", 60))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for t in dates:
        # Load GT alpha for date t
        gt_df = load_alpha_gt_for_date(Path(args.alpha_gt_path), t)
        if gt_df.empty:
            continue
        y = gt_df["alpha_z"]

        # Build features either from engine at t-1 or from logged columns (if present)
        if args.features_source == "logged" and {"mom_z_tm1","val_z_tm1"}.issubset(gt_df.columns):
            X = gt_df[["mom_z_tm1","val_z_tm1"]].rename(columns={"mom_z_tm1":"mom_z","val_z_tm1":"val_z"}).copy()
        else:
            X = build_engine_features(
                prices_df, fundamentals_df, sectors_ser, t,
                mom_lb=mom_lb, mom_gap=mom_gap, val_min_lag_days=val_lag,
                winsor_pcts=tuple(args.winsor_pcts) if args.winsor_pcts else None,
                sector_mode="neutralize" if args.sector_mode == "neutralize" else "none"
            )

        # Align indexes
        idx = X.index.intersection(y.index)
        X = X.reindex(idx).dropna()
        y = y.reindex(idx).dropna()
        if X.empty or y.empty or X.index.empty:
            continue

        # Optional "fixed effects": subtract sector means from BOTH features and alpha
        if args.sector_mode == "fixedeffects" and sectors_ser is not None:
            sec = sectors_ser.reindex(X.index)
            df_joint = pd.concat([X, y.rename("alpha_z")], axis=1)
            df_joint = sector_demean(df_joint, sec, cols=["mom_z","val_z","alpha_z"])
            X = df_joint[["mom_z","val_z"]]
            y = df_joint["alpha_z"]

        # Projection
        res = ols_capture(X, y, intercept=args.intercept)
        rows.append({
            "asof_dt": t,
            "r2": res["r2"],
            "n": res["n"],
            "beta_mom": res["betas"].get("mom_z", np.nan),
            "beta_val": res["betas"].get("val_z", np.nan),
            "intercept": res["intercept"],
            "features_source": args.features_source,
            "sector_mode": args.sector_mode,
            "winsor_low": (args.winsor_pcts[0] if args.winsor_pcts else np.nan),
            "winsor_high": (args.winsor_pcts[1] if args.winsor_pcts else np.nan),
        })

    if not rows:
        print("No capture rows computed (check alpha_gt coverage and features).", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows).sort_values("asof_dt")
    csv_path = outdir / f"alpha_capture_{dates[-1].date().isoformat()}.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    d_ok = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["r2"])
    r2_mean = float(d_ok["r2"].mean()) if not d_ok.empty else np.nan
    r2_med  = float(d_ok["r2"].median()) if not d_ok.empty else np.nan
    r2_p10  = float(d_ok["r2"].quantile(0.10)) if not d_ok.empty else np.nan
    r2_p90  = float(d_ok["r2"].quantile(0.90)) if not d_ok.empty else np.nan
    b_mom_mean = float(d_ok["beta_mom"].mean()) if not d_ok.empty else np.nan
    b_val_mean = float(d_ok["beta_val"].mean()) if not d_ok.empty else np.nan

    txt_path = outdir / f"alpha_capture_{dates[-1].date().isoformat()}.txt"
    with open(txt_path, "w") as f:
        f.write("ALPHA CAPTURE (PROJECTION) REPORT\n")
        f.write(f"Anchor: {dates[-1].date().isoformat()}  Window: {len(df)} days\n")
        f.write("="*60 + "\n")
        f.write(f"Features: {args.features_source}   Sector mode: {args.sector_mode}   Intercept: {args.intercept}\n")
        if args.winsor_pcts:
            f.write(f"Winsorization: [{args.winsor_pcts[0]:.3f}, {args.winsor_pcts[1]:.3f}]\n")
        f.write("\nSummary:\n")
        f.write(f"R^2 mean={r2_mean:.3f}, median={r2_med:.3f}, p10={r2_p10:.3f}, p90={r2_p90:.3f}, N_days={d_ok.shape[0]}\n")
        f.write(f"Mean betas: mom={b_mom_mean:.3f}, val={b_val_mean:.3f}\n")
        f.write("\nFirst/last 3 rows:\n")
        f.write(df.head(3).to_string(index=False) + "\n...\n" + df.tail(3).to_string(index=False) + "\n")

    print(f"Wrote {txt_path} and {csv_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 