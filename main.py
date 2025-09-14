from __future__ import annotations
def load_cern_root(path: str, tree: str, map_t: str, map_p: str, map_q: str, map_sigma: str = None) -> pd.DataFrame:
    """
    Load a CERN ROOT file and extract required columns as a DataFrame.

    Parameters
    ----------
    path : str
        Path to the ROOT file (local or remote).
    tree : str
        Name of the TTree to extract.
    map_t : str
        Branch name for proper time (t_ps) or flight distance (if using --derive_t).
    map_p : str
        Branch name for momentum (p_GeV).
    map_q : str
        Branch name for flavour tag.
    map_sigma : str, optional
        Branch name for time uncertainty (sigma_t_ps).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: t_ps, p_GeV, flavour_tag, sigma_t_ps (if available).
    """
    import uproot
    import awkward as ak
    with uproot.open(path) as f:
        ttree = f[tree]
        arrays = ttree.arrays([map_t, map_p, map_q] + ([map_sigma] if map_sigma else []), library="ak")
        df = ak.to_pandas(arrays)
        df = df.rename(columns={map_t: "t_ps", map_p: "p_GeV", map_q: "flavour_tag"})
        if map_sigma and map_sigma in df.columns:
            df = df.rename(columns={map_sigma: "sigma_t_ps"})
        return df

def load_cern_root_lhcb_b2hhh(path: str, tree: str = "DecayTree",
                              mass_GeV: float = 5.366,
                              units_mev: bool = True) -> pd.DataFrame:
    """
    Specialized loader for LHCb B2HHH training files.

    It extracts B_FlightDistance and daughter momentum components
    (H1/H2/H3 _PX/_PY/_PZ), computes the B momentum magnitude and
    derives t_ps = (L / c) * (m_B / p). Returns a DataFrame with
    columns `t_ps`, `p_GeV`, `flavour_tag` (if present), and
    `sigma_t_ps` if available.
    """
    import uproot
    import awkward as ak

    needed = [
        "B_FlightDistance",
        "H1_PX",
        "H1_PY",
        "H1_PZ",
        "H2_PX",
        "H2_PY",
        "H2_PZ",
        "H3_PX",
        "H3_PY",
        "H3_PZ",
    ]
    # optional branches
    opt = ["flavour_tag", "B_TIME_ERR_PS"]
    with uproot.open(path) as f:
        ttree = f[tree]
        available = set(ttree.keys())
        miss = [b for b in needed if b not in available]
        if miss:
            raise ValueError(f"Missing expected branches in file: {miss}")
        # prefer numpy-backed arrays to simplify conversion to pandas
        arrs = ttree.arrays(needed + [b for b in opt if b in available], library="np")

    import numpy as np
    df = pd.DataFrame({k: np.asarray(v) for k, v in arrs.items()})
    # compute B momentum from daughter components
    px = df["H1_PX"].fillna(0.0) + df["H2_PX"].fillna(0.0) + df["H3_PX"].fillna(0.0)
    py = df["H1_PY"].fillna(0.0) + df["H2_PY"].fillna(0.0) + df["H3_PY"].fillna(0.0)
    pz = df["H1_PZ"].fillna(0.0) + df["H2_PZ"].fillna(0.0) + df["H3_PZ"].fillna(0.0)
    p_mag = (px * px + py * py + pz * pz) ** 0.5
    # convert to GeV if stored in MeV
    if units_mev:
        p_GeV = p_mag / 1000.0
    else:
        p_GeV = p_mag
    # derive proper time in ps: t = (L / c) * (m / p)
    c_mm_ps = 299.792
    L = df["B_FlightDistance"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ps = (L / c_mm_ps) * (mass_GeV / p_GeV)
    out = pd.DataFrame({
        "t_ps": t_ps.astype(float),
        "p_GeV": p_GeV.astype(float),
    })
    if "flavour_tag" in df.columns:
        out["flavour_tag"] = df["flavour_tag"].astype(float)
    else:
        out["flavour_tag"] = 0.0
    if "B_TIME_ERR_PS" in df.columns:
        out["sigma_t_ps"] = df["B_TIME_ERR_PS"].astype(float)
    return out
import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt


@dataclass
class FitResult:
    """Fit results for a single gamma bin.

    Attributes:
        params_baseline: MLE parameters for the baseline model.
        nll_baseline: Negative log-likelihood for baseline.
        params_extended: MLE parameters for the extended model.
        nll_extended: Negative log-likelihood for extended.
        lambda_hat: Fitted extra damping (ps^-1).
        gamma_mean: Mean gamma in the bin.
        gamma_lo: Lower gamma edge.
        gamma_hi: Upper gamma edge.
        n_events: Number of events in the bin.
    """
    params_baseline: Dict[str, float]
    nll_baseline: float
    params_extended: Dict[str, float]
    nll_extended: float
    lambda_hat: float
    gamma_mean: float
    gamma_lo: float
    gamma_hi: float
    n_events: int


def relativistic_gamma(p_GeV: np.ndarray, mass_GeV: float) -> np.ndarray:
    """Return gamma = sqrt(p^2 + m^2)/m for momentum p and mass m."""
    e = np.sqrt(p_GeV * p_GeV + mass_GeV * mass_GeV)
    return e / mass_GeV


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return log with lower clipping to avoid -inf."""
    return np.log(np.clip(x, eps, None))


def oscillation_pdf(
    t_ps: np.ndarray,
    q: np.ndarray,
    A0: float,
    dm: float,
    Gamma: float,
    lam_extra: float = 0.0,
) -> np.ndarray:
    """Return normalized PDF on t≥0: exp(-(Γ+λ)t)*(1+q*A0*cos(Δm t))."""
    alpha = Gamma + lam_extra
    osc = 1.0 + q * A0 * np.cos(dm * t_ps)
    expo = np.exp(-alpha * t_ps)
    u = np.clip(expo * osc, 1e-300, None)
    z = (1.0 / alpha) + q * A0 * (alpha / (alpha * alpha + dm * dm))
    z = np.clip(z, 1e-300, None)
    return np.clip(u / z, 1e-300, None)


def convolve_time_resolution(
    t_ps: np.ndarray,
    pdf_fn: Callable[[np.ndarray], np.ndarray],
    sigma_ps: Optional[np.ndarray],
    n_sigma: float = 4.0,
    n_grid: int = 9,
) -> np.ndarray:
    """Approximate Gaussian resolution convolution via fixed grid."""
    if sigma_ps is None or np.all(sigma_ps <= 0.0):
        return pdf_fn(t_ps)
    zs = np.linspace(-n_sigma, n_sigma, n_grid)
    ws = norm.pdf(zs)
    ws /= np.sum(ws)
    vals = np.zeros_like(t_ps, dtype=float)
    for z, w in zip(zs, ws):
        vals += w * pdf_fn(np.clip(t_ps + z * sigma_ps, 0.0, None))
    return vals


def neg_loglike(
    x: np.ndarray,
    t_ps: np.ndarray,
    q: np.ndarray,
    sigma_t_ps: Optional[np.ndarray],
    model: str,
    fixed: Dict[str, float],
) -> float:
    """Return negative log-likelihood for baseline or extended model."""
    A0 = x[0]
    idx = 1
    if "dm" in fixed:
        dm = fixed["dm"]
    else:
        dm = x[idx]
        idx += 1
    if "Gamma" in fixed:
        Gamma = fixed["Gamma"]
    else:
        Gamma = x[idx]
        idx += 1
    lam_extra = 0.0
    if model == "extended":
        lam_extra = x[idx]

    def core_pdf(tv: np.ndarray) -> np.ndarray:
        return oscillation_pdf(
            tv, q, A0=A0, dm=dm, Gamma=Gamma, lam_extra=lam_extra
        )

    pdf_vals = convolve_time_resolution(
        t_ps=t_ps, pdf_fn=core_pdf, sigma_ps=sigma_t_ps
    )
    return float(-np.sum(safe_log(pdf_vals)))


def fit_bin_mle(
    t_ps: np.ndarray,
    q: np.ndarray,
    sigma_t_ps: Optional[np.ndarray],
    init: Dict[str, float],
    fixed: Dict[str, float],
    model: str,
) -> Tuple[Dict[str, float], float]:
    """Fit one gamma bin by MLE and return parameters with NLL."""
    x0: List[float] = [init.get("A0", 0.6)]
    bounds: List[Tuple[float, float]] = [(-0.999, 0.999)]
    # Only add free parameters to x0 and bounds
    if "dm" not in fixed:
        x0.append(init.get("dm", 17.7))
        bounds.append((0.01, 40.0))
    if "Gamma" not in fixed:
        x0.append(init.get("Gamma", 0.66))
        bounds.append((0.05, 5.0))
    if model == "extended":
        x0.append(init.get("lambda", 0.001))
        bounds.append((0.0, 2.0))

    def fun(xx: np.ndarray) -> float:
        return neg_loglike(xx, t_ps, q, sigma_t_ps, model=model, fixed=fixed)

    res = minimize(fun, np.array(x0), method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"Fit failed: {res.message}")
    out = {"A0": float(res.x[0])}
    idx = 1
    if "dm" in fixed:
        out["dm"] = float(fixed["dm"])
    else:
        out["dm"] = float(res.x[idx]); idx += 1
    if "Gamma" in fixed:
        out["Gamma"] = float(fixed["Gamma"])
    else:
        out["Gamma"] = float(res.x[idx]); idx += 1
    if model == "extended":
        out["lambda"] = float(res.x[idx])
    return out, float(res.fun)


def run_analysis(
    df: pd.DataFrame,
    out_dir: str,
    mass_GeV: float,
    n_gbins: int,
    dm_hint: float,
    Gamma_hint: float,
    fix_dm: bool,
    fix_Gamma: bool,
) -> List[FitResult]:
    """Bin by gamma, fit both models, test trend, and write outputs."""
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    df["gamma"] = relativistic_gamma(df["p_GeV"].values, mass_GeV)
    msk = (
        (df["t_ps"] >= 0.0)
        & np.isfinite(df["t_ps"])  
        & np.isfinite(df["gamma"])  
        & np.isfinite(df["p_GeV"])  
        & np.isfinite(df.get("flavour_tag", 0))
    )
    df = df.loc[msk].reset_index(drop=True)
    qs = np.linspace(0.0, 1.0, n_gbins + 1)
    edges = df["gamma"].quantile(qs).values
    edges[0] = min(edges[0], 1.0)
    edges[-1] = max(edges[-1], float(df["gamma"].max()))
    results: List[FitResult] = []
    for i in range(n_gbins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < n_gbins - 1:
            subi = df[(df["gamma"] >= lo) & (df["gamma"] < hi)]
        else:
            subi = df[(df["gamma"] >= lo) & (df["gamma"] <= hi)]
        if len(subi) < 500:
            continue
        t = subi["t_ps"].values.astype(float)
        q = subi.get("flavour_tag", pd.Series(np.zeros(len(subi)))).values
        sigma = subi.get("sigma_t_ps", pd.Series(np.zeros(len(subi)))).values
        fixed: Dict[str, float] = {}
        if fix_dm:
            fixed["dm"] = float(dm_hint)
        if fix_Gamma:
            fixed["Gamma"] = float(Gamma_hint)
        init = {
            "A0": 0.5,
            "dm": float(dm_hint),
            "Gamma": float(Gamma_hint),
            "lambda": 0.001,
        }
        pb, nll_b = fit_bin_mle(t, q, sigma, init, fixed, model="baseline")
        pe, nll_e = fit_bin_mle(t, q, sigma, init, fixed, model="extended")
        lam_hat = float(pe.get("lambda", 0.0))
        res = FitResult(
            params_baseline=pb,
            nll_baseline=float(nll_b),
            params_extended=pe,
            nll_extended=float(nll_e),
            lambda_hat=lam_hat,
            gamma_mean=float(subi["gamma"].mean()),
            gamma_lo=lo,
            gamma_hi=hi,
            n_events=int(len(subi)),
        )
        results.append(res)
    if not results:
        return results
    gm = np.array([r.gamma_mean for r in results])
    lam = np.array([r.lambda_hat for r in results])
    dchi = np.array(
        [max(0.0, 2.0 * (r.nll_extended - r.nll_baseline)) for r in results]
    )
    se = np.where(
        dchi > 1e-6,
        np.abs(lam) / np.sqrt(dchi + 1e-9),
        np.maximum(0.02, 0.05 * np.abs(lam) + 0.02),
    )
    g0 = gm - gm.mean()
    w = 1.0 / np.maximum(se * se, 1e-6)
    b_hat = float(np.sum(w * g0 * lam) / np.sum(w * g0 * g0))
    a_hat = float(np.average(lam - b_hat * g0, weights=w))
    b_se = float(1.0 / math.sqrt(np.sum(w * g0 * g0)))
    z = b_hat / (b_se + 1e-12)
    plt.figure()
    plt.errorbar(gm, lam, yerr=se, fmt="o", capsize=3)
    xs = np.linspace(gm.min(), gm.max(), 100)
    ys = a_hat + b_hat * (xs - gm.mean())
    plt.plot(xs, ys, "-", label=f"slope={b_hat:.4g}±{b_se:.4g}, z={z:.2f}")
    plt.xlabel("mean γ (bin)")
    plt.ylabel("λ̂ (ps⁻¹)")
    plt.title("Extended damping vs gamma")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        os.path.join(out_dir, "lambda_vs_gamma.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()
    lines: List[str] = []
    lines.append("Per-bin fit summary (gamma bins):\n")
    for r in results:
        dchi_i = 2.0 * (r.nll_extended - r.nll_baseline)
        lines.append(
            f"γ∈[{r.gamma_lo:.3g},{r.gamma_hi:.3g}]  "
            f"⟨γ⟩={r.gamma_mean:.3g}  N={r.n_events:6d}  "
            f"λ̂={r.lambda_hat:.5g} ps^-1  "
            f"2ΔNLL={dchi_i:.3g}  "
            f"A0={r.params_extended['A0']:.3g}  "
            f"dm={r.params_extended['dm']:.3g}  "
            f"Γ={r.params_extended['Gamma']:.3g}"
        )
    lines.append("\nTrend test (λ vs γ):")
    lines.append(
        f"slope={b_hat:.6g} ± {b_se:.6g}, z={z:.3f} (|z|>3 ≈ strong evidence)"
    )
    with open(os.path.join(out_dir, "fit_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return results


def load_dataframe(path: str) -> pd.DataFrame:
    """Load CSV or Parquet; Parquet requires an installed engine."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise ValueError(
                "Parquet engine missing. Install pyarrow/fastparquet or convert to CSV"
            ) from e
    raise ValueError("Unsupported file extension; use .csv or .parquet")


def generate_demo_dataframe(
    n_events: int = 30000,
    mass_GeV: float = 5.366,
    dm: float = 17.77,
    Gamma: float = 0.66,
    lam0: float = 0.0,
    lam_slope: float = 0.0,
    seed: int = 123,
) -> pd.DataFrame:
    """Return synthetic dataset for quick tests."""
    rng = np.random.default_rng(seed)
    p = rng.gamma(shape=2.0, scale=6.0, size=n_events)
    gamma = relativistic_gamma(p, mass_GeV)
    u = rng.uniform(size=n_events)
    q = np.where(u < 0.45, 1.0, np.where(u < 0.9, -1.0, 0.0))
    g0 = gamma - gamma.mean()
    lam = np.clip(lam0 + lam_slope * g0, 0.0, None)
    rate = Gamma + lam
    t = rng.exponential(scale=1.0 / np.clip(rate, 1e-6, None))
    sigma = rng.normal(loc=0.02, scale=0.01, size=n_events)
    sigma = np.clip(sigma, 0.0, None)
    return pd.DataFrame({
        "t_ps": t.astype(float),
        "p_GeV": p.astype(float),
        "flavour_tag": q.astype(float),
        "sigma_t_ps": sigma.astype(float),
    })


def selftest_quick(out_dir: str) -> None:
    """Smoke test: expects near-zero slope on λ vs γ."""
    df = generate_demo_dataframe(n_events=15000, lam0=0.0, lam_slope=0.0)
    run_analysis(
        df=df,
        out_dir=out_dir,
        mass_GeV=5.366,
        n_gbins=5,
        dm_hint=17.77,
        Gamma_hint=0.66,
        fix_dm=True,
        fix_Gamma=True,
    )


def selftest_full(out_dir: str) -> None:
    """End-to-end test: positive γ-trend in λ."""
    df = generate_demo_dataframe(n_events=40000, lam0=0.0, lam_slope=0.02)
    run_analysis(
        df=df,
        out_dir=out_dir,
        mass_GeV=5.366,
        n_gbins=6,
        dm_hint=17.77,
        Gamma_hint=0.66,
        fix_dm=True,
        fix_Gamma=True,
    )


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to CSV/Parquet with required cols")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--mass_GeV", type=float, default=5.366)
    ap.add_argument("--n_gbins", type=int, default=6)
    ap.add_argument("--dm_hint", type=float, default=17.77)
    ap.add_argument("--Gamma_hint", type=float, default=0.66)
    ap.add_argument("--fix_dm", action="store_true")
    ap.add_argument("--fix_Gamma", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--selftest", choices=["quick", "full"]) 
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(
        level=(
            logging.WARNING
            if args.verbosity <= 0
            else logging.INFO
            if args.verbosity == 1
            else logging.DEBUG
        )
    )

    if args.selftest == "quick":
        selftest_quick(out_dir=os.path.join(args.out, "selftest_quick"))
        print(f"Selftest quick done → {os.path.join(args.out, 'selftest_quick')}")
        return
    if args.selftest == "full":
        selftest_full(out_dir=os.path.join(args.out, "selftest_full"))
        print(f"Selftest full done → {os.path.join(args.out, 'selftest_full')}")
        return

    if args.data is None:
        df = generate_demo_dataframe()
    else:
        df = load_dataframe(args.data)

    required = ["t_ps", "p_GeV", "flavour_tag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    res = run_analysis(
        df=df,
        out_dir=args.out,
        mass_GeV=args.mass_GeV,
        n_gbins=args.n_gbins,
        dm_hint=args.dm_hint,
        Gamma_hint=args.Gamma_hint,
        fix_dm=args.fix_dm,
        fix_Gamma=args.fix_Gamma,
    )

    if not res:
        print("No bins had sufficient events. Try fewer bins or more data.")
    else:
        print(f"Completed. Wrote report and plot to: {args.out}")


if __name__ == "__main__":
    main()
