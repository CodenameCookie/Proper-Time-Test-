import argparse
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse
import requests
import logging
import pandas as pd
from main import run_analysis, load_dataframe

def _download_to_temp(url: str, suffix: str | None = None) -> Path:
    """
    Download a file to a secure temp location.

    Parameters
    ----------
    url : str
        HTTP(S) URL of the remote file.
    suffix : Optional[str]
        File extension hint (e.g., ".csv", ".parquet", ".root").

    Returns
    -------
    pathlib.Path
        Path to the temporary file on disk.

    Raises
    ------
    requests.HTTPError
        If the HTTP response status is not 200.
    requests.RequestException
        For connection issues/timeouts.
    """
    timeout = (10, 10)
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    guessed = Path(urlparse(url).path).suffix
    ext = suffix or guessed or ""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    p = Path(tf.name)
    with tf:
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            tf.write(chunk)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to CSV/Parquet with required cols")
    ap.add_argument("--remote_url",
                    help="HTTP(S) URL to CSV/Parquet/ROOT file to fetch")
    ap.add_argument("--preset", choices=["lhcb_b2hhh"],
                    help="Auto-map LHCb B2HHH ROOT files (DecayTree, B_TIME_PS, B_P_GeV, flavour_tag, B_TIME_ERR_PS)")
    ap.add_argument("--derive_t", action="store_true",
                    help="Derive t_ps from flight distance and momentum if direct time is missing.")
    ap.add_argument("--map_L", help="ROOT branch for flight distance (mm)")
    ap.add_argument("--map_m", type=float, default=5.366,
                    help="Mass of B meson in GeV/c^2 (default: 5.366)")
    ap.add_argument("--out", default="out_csv", help="Output directory")
    ap.add_argument("--mass_GeV", type=float, default=5.366)
    ap.add_argument("--n_gbins", type=int, default=6)
    ap.add_argument("--dm_hint", type=float, default=17.77)
    ap.add_argument("--Gamma_hint", type=float, default=0.66)
    ap.add_argument("--fix_dm", action="store_true")
    ap.add_argument("--fix_Gamma", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--selftest", choices=["quick", "full"])
    # ROOT support flags (preserved for context)
    ap.add_argument("--root", help="Path to CERN ROOT file")
    ap.add_argument("--tree", help="ROOT tree name")
    ap.add_argument("--map_t", help="ROOT branch for t_ps")
    ap.add_argument("--map_p", help="ROOT branch for p_GeV")
    ap.add_argument("--map_q", help="ROOT branch for flavour_tag")
    ap.add_argument("--map_sigma", help="ROOT branch for sigma_t_ps")
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument(
        "--allow_untagged",
        action="store_true",
        help="If set, proceed when flavour_tag is missing by treating all events as untagged (q=0).",
    )
    ap.add_argument(
        "--infer_tag_from_charges",
        action="store_true",
        help=(
            "Try to infer a flavour tag from charge columns (e.g. H1_Charge,H2_Charge). "
            "Provide names with --map_charge_cols if automatic detection fails."
        ),
    )
    ap.add_argument(
        "--map_charge_cols",
        help=(
            "Comma-separated list of ROOT/Pandas column names to use for tag inference. "
            "Example: --map_charge_cols H1_Charge,H2_Charge"
        ),
    )
    ap.add_argument(
        "--p_max",
        type=float,
        default=None,
        help="Clip p_GeV to this absolute maximum to guard against extreme outliers.",
    )
    ap.add_argument(
        "--p_clip_quantile",
        type=float,
        default=None,
        help="Clip p_GeV at this upper quantile (0-1) to remove extreme tails.",
    )
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

    # Loader precedence: selftest/demo > remote_url > root > data > fallback demo
    if args.selftest == "quick":
        # ...existing code...
        return

    if args.selftest == "full":
        # ...existing code...
        return

    if args.demo:
        from main import generate_demo_dataframe
        df = generate_demo_dataframe()
    elif args.remote_url:
        url = args.remote_url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("remote_url must be http(s)://")
        tmp_path = _download_to_temp(url)
        ext = Path(tmp_path).suffix.lower()
        if ext == ".root":
            # For LHCB preset, use the specialized loader that derives
            # p_GeV from daughter momenta and computes t_ps from
            # B_FlightDistance.
            if args.preset == "lhcb_b2hhh":
                from main import load_cern_root_lhcb_b2hhh

                df = load_cern_root_lhcb_b2hhh(
                    path=str(tmp_path),
                    tree="DecayTree",
                    mass_GeV=args.map_m,
                    units_mev=True,
                )
            else:
                from main import load_cern_root

                if args.tree and args.map_t and args.map_p and args.map_q:
                    df = load_cern_root(
                        path=str(tmp_path),
                        tree=args.tree,
                        map_t=args.map_t,
                        map_p=args.map_p,
                        map_q=args.map_q,
                        map_sigma=args.map_sigma,
                    )
                else:
                    raise ValueError(
                        "ROOT input from URL requires --tree, --map_t, --map_p, "
                        "--map_q (and optionally --map_sigma), or use --preset lhcb_b2hhh."
                    )
        elif ext in (".csv", ".parquet"):
            df = load_dataframe(str(tmp_path))
        else:
            raise ValueError(
                f"Unsupported remote extension '{ext}'. Use CSV, Parquet, or ROOT."
            )
    elif args.root:
        if not args.tree or not args.map_t or not args.map_p or not args.map_q:
            raise ValueError("--root requires --tree, --map_t, --map_p, --map_q")
        from main import load_cern_root
        df = load_cern_root(
            path=args.root,
            tree=args.tree,
            map_t=args.map_t,
            map_p=args.map_p,
            map_q=args.map_q,
            map_sigma=args.map_sigma,
        )
    elif args.data:
        try:
            df = load_dataframe(args.data)
        except Exception as e:
            print(f"Error loading data: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        from main import generate_demo_dataframe
        df = generate_demo_dataframe()

    logging.info("DataFrame loaded: %d rows, cols=%s",
                 len(df), list(df.columns))

    # Optional: clip p_GeV outliers
    if args.p_max is not None:
        if "p_GeV" in df.columns:
            df["p_GeV"] = df["p_GeV"].clip(upper=float(args.p_max))
            logging.info("Clipped p_GeV to max=%s", args.p_max)
    if args.p_clip_quantile is not None:
        q = float(args.p_clip_quantile)
        if 0.0 < q < 1.0 and "p_GeV" in df.columns:
            um = df["p_GeV"].quantile(q)
            df["p_GeV"] = df["p_GeV"].clip(upper=um)
            logging.info("Clipped p_GeV at quantile %.3g -> %.6g", q, um)

    # If --derive_t, compute t_ps from flight distance and momentum
    if args.derive_t:
        if "t_ps" not in df.columns:
            if args.map_L and "p_GeV" in df.columns:
                # c = 299.792 mm/ps
                c_mm_ps = 299.792
                m_GeV = args.map_m
                if args.map_L not in df.columns:
                    print(f"Missing flight distance column: {args.map_L}", file=sys.stderr)
                    sys.exit(1)
                L = df[args.map_L].astype(float)
                p = df["p_GeV"].astype(float)
                df["t_ps"] = (L / c_mm_ps) * (m_GeV / p)
                logging.info("Derived t_ps from L and p_GeV using m_B=%.4f GeV/c^2", m_GeV)
            else:
                print("--derive_t requires --map_L and p_GeV column in data.", file=sys.stderr)
                sys.exit(1)

    # Handle flavour tag presence / inference / untagged mode
    if "flavour_tag" not in df.columns:
        if args.infer_tag_from_charges:
            # try to infer from provided map_charge_cols or common names
            candidates = []
            if args.map_charge_cols:
                candidates = [c.strip() for c in args.map_charge_cols.split(",") if c.strip()]
            else:
                # common LHCb names
                candidates = [c for c in ["H1_Charge", "H2_Charge", "H3_Charge"] if c in df.columns]
            if candidates:
                # simple rule: sum charges of candidates; positive -> +1, negative -> -1, zero -> 0
                try:
                    chsum = df[candidates].sum(axis=1)
                    df["flavour_tag"] = chsum.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
                    logging.info("Inferred flavour_tag from charge cols: %s", candidates)
                except Exception as e:
                    logging.warning("Failed to infer flavour_tag from %s: %s", candidates, e)
            else:
                logging.info("No charge columns found to infer flavour_tag")
        if "flavour_tag" not in df.columns:
            if args.allow_untagged:
                df["flavour_tag"] = 0.0
                logging.info("Proceeding with untagged data (flavour_tag=0)")
            else:
                print("Missing required column: flavour_tag. Use --allow_untagged or --infer_tag_from_charges.", file=sys.stderr)
                sys.exit(1)

    # final required check for t_ps and p_GeV
    required = ["t_ps", "p_GeV"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    results = run_analysis(
        df=df,
        out_dir=args.out,
        mass_GeV=args.mass_GeV,
        n_gbins=args.n_gbins,
        dm_hint=args.dm_hint,
        Gamma_hint=args.Gamma_hint,
        fix_dm=args.fix_dm,
        fix_Gamma=args.fix_Gamma,
    )

    if not results:
        print("No bins had sufficient events. Try fewer bins or more data.")
        return

    print(f"Analysis complete. Results written to: {args.out}")
    print(f"Number of gamma bins: {len(results)}")
    print("Events per bin:")
    for i, r in enumerate(results):
        print(f"  Bin {i+1}: N={r.n_events}, gamma=[{r.gamma_lo:.3g}, "
              f"{r.gamma_hi:.3g}], lambda_hat={r.lambda_hat:.5g}")
    report_path = os.path.join(args.out, "fit_report.txt")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip().startswith("slope="):
                print("Trend test:", line.strip())
                break

if __name__ == "__main__":
    main()
