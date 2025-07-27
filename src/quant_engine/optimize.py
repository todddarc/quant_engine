"""
Portfolio optimization module for mean-variance optimization.

Implements mean-variance optimization with constraints including long-only,
per-name caps, sector caps, and turnover limits.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from scipy.optimize import minimize
import logging


def mean_variance_opt(
    alpha: pd.Series,
    Sigma: pd.DataFrame,
    sectors_map: Union[pd.Series, dict],
    prev_w: Union[pd.Series, None],
    *,
    long_only: bool = True,
    w_max: float = 0.05,
    sector_cap: float = 0.25,
    turnover_cap: float = 0.10,
    risk_aversion: float = 8.0,
    eps_turnover: float = 1e-8,
    max_iter: int = 500,
    fixed_weights: Optional[pd.Series] = None,
) -> Tuple[pd.Series, Dict[str, object]]:
    """
    Mean–variance portfolio optimization:
        maximize   w^T alpha - (risk_aversion/2) * w^T Sigma w
        subject to sum(w)=1, 0<=w<=w_max, sum_{i in sector s} w_i <= sector_cap, 
                   0.5 * sum_i |w_i - prev_w_i| <= turnover_cap   (smooth abs)
    
    Args:
        alpha: Expected returns series indexed by ticker
        Sigma: Covariance matrix DataFrame with matching ticker index/columns
        sectors_map: Mapping from ticker to sector (Series or dict)
        prev_w: Prior weights series (can be None)
        long_only: Whether to enforce long-only constraint
        w_max: Maximum weight per asset
        sector_cap: Maximum weight per sector
        turnover_cap: Maximum turnover constraint
        risk_aversion: Risk aversion parameter
        eps_turnover: Smoothing parameter for turnover constraint
        max_iter: Maximum iterations for optimizer
        fixed_weights: Optional Series indexed by tickers to be held fixed at the given values 
                      (typically prev_w for frozen names). When provided, for any i in 
                      fixed_weights.index, set bounds[i] = (fixed_weights[i], fixed_weights[i]).
        
    Returns:
        Tuple of (weights Series indexed by ticker, diagnostics dict)
        
    Notes:
        - Sigma must be a pd.DataFrame with index/columns matching the ticker order of alpha
        - Uses smooth absolute value: abs(x) ≈ sqrt(x^2 + eps_turnover)
        - If optimizer fails or problem appears infeasible, returns prev_w and diagnostics["success"]=False
        - Fixed weights contribute zero turnover automatically since w_i == prev_w_i
    """
    # Align tickers across alpha and Sigma
    common_tickers = alpha.index.intersection(Sigma.index)
    if len(common_tickers) < 2:
        logging.warning(f"mean_variance_opt: insufficient tickers ({len(common_tickers)})")
        if prev_w is not None:
            fallback_w = prev_w.reindex(common_tickers).fillna(1.0/len(common_tickers))
        else:
            fallback_w = pd.Series(1.0/len(common_tickers), index=common_tickers)
        return fallback_w, {"success": False, "message": "Insufficient tickers"}
    
    # Align data to common tickers
    alpha_aligned = alpha.loc[common_tickers]
    Sigma_aligned = Sigma.loc[common_tickers, common_tickers]
    
    # Align sectors_map
    if isinstance(sectors_map, dict):
        sectors_series = pd.Series(sectors_map)
    else:
        sectors_series = sectors_map.copy()
    
    sectors_aligned = sectors_series.reindex(common_tickers)
    tickers_with_sectors = sectors_aligned.dropna().index
    
    if len(tickers_with_sectors) < 2:
        logging.warning(f"mean_variance_opt: insufficient tickers with sectors ({len(tickers_with_sectors)})")
        if prev_w is not None:
            fallback_w = prev_w.reindex(common_tickers).fillna(1.0/len(common_tickers))
        else:
            fallback_w = pd.Series(1.0/len(common_tickers), index=common_tickers)
        return fallback_w, {"success": False, "message": "Insufficient tickers with sectors"}
    
    # Use only tickers with sectors
    alpha_opt = alpha_aligned.loc[tickers_with_sectors]
    Sigma_opt = Sigma_aligned.loc[tickers_with_sectors, tickers_with_sectors]
    sectors_opt = sectors_aligned.loc[tickers_with_sectors]
    
    # Align prev_w if provided
    prev_w_aligned = None
    if prev_w is not None:
        prev_w_aligned = prev_w.reindex(tickers_with_sectors).fillna(1.0/len(tickers_with_sectors))
    
    # Align fixed_weights if provided
    fixed_weights_aligned = None
    if fixed_weights is not None:
        fixed_weights_aligned = fixed_weights.reindex(tickers_with_sectors).dropna()
        if len(fixed_weights_aligned) == 0:
            fixed_weights_aligned = None
        else:
            logging.info(f"mean_variance_opt: {len(fixed_weights_aligned)} tickers fixed at their weights")
    
    n_assets = len(tickers_with_sectors)
    
    # Convert to numpy arrays
    alpha_np = alpha_opt.values
    Sigma_np = Sigma_opt.values
    
    # Define objective function: minimize -f(w) where f(w) = w^T alpha - 0.5 * risk_aversion * w^T Sigma w
    def objective(w):
        return -(w @ alpha_np - 0.5 * risk_aversion * w @ Sigma_np @ w)
    
    # Define gradient: ∇f(w) = -alpha + risk_aversion * Sigma @ w
    def gradient(w):
        return -alpha_np + risk_aversion * Sigma_np @ w
    
    # Define turnover function and gradient
    def turnover_func(w):
        if prev_w_aligned is None:
            return 0.0
        diff = w - prev_w_aligned.values
        return 0.5 * np.sum(np.sqrt(diff**2 + eps_turnover))
    
    def turnover_gradient(w):
        if prev_w_aligned is None:
            return np.zeros_like(w)
        diff = w - prev_w_aligned.values
        return 0.5 * diff / np.sqrt(diff**2 + eps_turnover)
    
    # Create constraints
    constraints = []
    
    # Equality constraint: sum(w) = 1
    constraints.append({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1.0,
        'jac': lambda w: np.ones(n_assets)
    })
    
    # Sector cap constraints
    unique_sectors = sectors_opt.unique()
    for sector in unique_sectors:
        sector_mask = sectors_opt == sector
        sector_indices = np.where(sector_mask)[0]
        
        def sector_constraint(w, sector_idx=sector_indices):
            return sector_cap - np.sum(w[sector_idx])
        
        def sector_jacobian(w, sector_idx=sector_indices):
            jac = np.zeros(n_assets)
            jac[sector_idx] = -1.0
            return jac
        
        constraints.append({
            'type': 'ineq',
            'fun': sector_constraint,
            'jac': sector_jacobian
        })
    
    # Turnover constraint
    if prev_w_aligned is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: turnover_cap - turnover_func(w),
            'jac': lambda w: -turnover_gradient(w)
        })
    
    # Bounds
    if long_only:
        bounds = [(0.0, w_max)] * n_assets
    else:
        bounds = [(-w_max, w_max)] * n_assets
    
    # Apply fixed weights constraints
    if fixed_weights_aligned is not None:
        for ticker, fixed_weight in fixed_weights_aligned.items():
            idx = tickers_with_sectors.get_loc(ticker)
            bounds[idx] = (fixed_weight, fixed_weight)
    
    # Initial guess
    if prev_w_aligned is not None:
        x0 = prev_w_aligned.values.copy()
        # Clip to bounds and renormalize
        x0 = np.clip(x0, 0.0, w_max)
        x0 = x0 / np.sum(x0)
    else:
        x0 = np.ones(n_assets) / n_assets
    
    # Overwrite fixed weights in initial guess
    if fixed_weights_aligned is not None:
        for ticker, fixed_weight in fixed_weights_aligned.items():
            idx = tickers_with_sectors.get_loc(ticker)
            x0[idx] = fixed_weight
    
    # Optimize
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-9, 'disp': False}
        )
        
        # Check success and constraint violations
        success = result.success
        logging.info(f"mean_variance_opt: optimizer success = {success}, message = {result.message}")
        
        # Check constraint violations
        w_opt = result.x
        
        # Check sum to one
        sum_violation = abs(np.sum(w_opt) - 1.0)
        if sum_violation > 1e-6:
            logging.warning(f"mean_variance_opt: sum violation = {sum_violation}")
            success = False
        
        # Check bounds
        bounds_violation = np.any((w_opt < -1e-6) | (w_opt > w_max + 1e-6))
        if bounds_violation:
            logging.warning(f"mean_variance_opt: bounds violation detected")
            success = False
        
        # Check sector caps
        sector_violations = 0
        for sector in unique_sectors:
            sector_mask = sectors_opt == sector
            sector_weight = np.sum(w_opt[sector_mask])
            if sector_weight > sector_cap + 1e-6:
                sector_violations += 1
                logging.warning(f"mean_variance_opt: sector {sector} violation = {sector_weight}")
                success = False
        
        # Check turnover
        if prev_w_aligned is not None:
            actual_turnover = 0.5 * np.sum(np.abs(w_opt - prev_w_aligned.values))
            if actual_turnover > turnover_cap + 1e-6:
                logging.warning(f"mean_variance_opt: turnover violation = {actual_turnover}")
                success = False
        
        # Check for NaNs
        if np.any(np.isnan(w_opt)):
            logging.warning(f"mean_variance_opt: NaN detected in weights")
            success = False
        
        if success:
            # Create weights series
            weights = pd.Series(w_opt, index=tickers_with_sectors)
            
            # Calculate diagnostics
            risk = float(w_opt @ Sigma_np @ w_opt)
            alpha_dot = float(alpha_np @ w_opt)
            
            diagnostics = {
                "success": True,
                "message": str(result.message),
                "obj": float(result.fun),
                "risk": risk,
                "alpha_dot": alpha_dot,
                "turnover": actual_turnover if prev_w_aligned is not None else float("nan"),
                "active_sector_violations": sector_violations
            }
            
            logging.info(f"mean_variance_opt: success, risk={risk:.6f}, alpha_dot={alpha_dot:.6f}")
            
        else:
            # Fallback to prev_w or equal weights
            if prev_w_aligned is not None:
                weights = prev_w_aligned
            else:
                weights = pd.Series(1.0/n_assets, index=tickers_with_sectors)
            
            diagnostics = {
                "success": False,
                "message": f"Optimization failed or constraints violated: {result.message}",
                "obj": float("nan"),
                "risk": float("nan"),
                "alpha_dot": float("nan"),
                "turnover": float("nan"),
                "active_sector_violations": sector_violations
            }
            
            logging.warning(f"mean_variance_opt: fallback to prior weights")
            
    except Exception as e:
        # Fallback to prev_w or equal weights
        if prev_w_aligned is not None:
            weights = prev_w_aligned
        else:
            weights = pd.Series(1.0/n_assets, index=tickers_with_sectors)
        
        diagnostics = {
            "success": False,
            "message": f"Optimization exception: {str(e)}",
            "obj": float("nan"),
            "risk": float("nan"),
            "alpha_dot": float("nan"),
            "turnover": float("nan"),
            "active_sector_violations": 0
        }
        
        logging.error(f"mean_variance_opt: exception, fallback to prior weights")
    
    return weights, diagnostics 