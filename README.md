# Proper-Time-Test-

Overview
--------
This repository contains a small analysis pipeline for studying an additional exponential damping
term (λ) in time-dependent oscillations of B-hadron decays as a function of boost (γ).

The core analysis lives in `main.py` and implements:
- data generation (demo/selftest)
- ROOT/CSV/Parquet loading helpers
- per-bin maximum-likelihood fits of a time PDF with an optional extra damping parameter λ
- a trend test to look for λ vs γ dependence

`gamma_lambda_fit.py` is a user-facing CLI wrapper that adds convenience for loading remote
datasets (HTTP/ROOT), LHCb presets, and a few safety/diagnostic flags.

Contents
--------
- `main.py` — core analysis functions and CLI (also used programmatically)
- `gamma_lambda_fit.py` — wrapper CLI for convenience, remote download, LHCb preset
- `out_*/` — example output directories created by runs

Requirements
------------
- Python 3.10+ (this workspace used 3.12 in a virtualenv)
- Minimal Python packages: numpy, pandas, scipy, matplotlib, requests
- Optional/for ROOT files: uproot, awkward
- For remote ROOT streaming: aiohttp / fsspec (if you stream directly)

Install (recommended: virtualenv)

1. Create and activate a venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you expect to read CERN ROOT files directly, install:

```bash
pip install uproot awkward aiohttp fsspec
```

Quick usage
-----------
Run the project wrapper (recommended):

```bash
python gamma_lambda_fit.py --demo --out out_demo
```

This runs a synthetic dataset and writes `fit_report.txt` and `lambda_vs_gamma.png` under `out_demo/`.

Loading remote data (ROOT) with LHCb preset
-----------------------------------------
The script supports downloading remote files via `--remote_url`. For LHCb B2HHH files there is a
convenience preset `--preset lhcb_b2hhh` which auto-maps daughter momentum components and
derives `p_GeV` (sum of daughters) and `t_ps` (from `B_FlightDistance`):

```bash
python gamma_lambda_fit.py \
	--remote_url http://.../B2HHH_MagnetDown.root \
	--preset lhcb_b2hhh \
	--out out_remote_root --fix_dm --fix_Gamma
```

Important CLI flags (wrapper `gamma_lambda_fit.py`)
-------------------------------------------------
- `--data <path>`: local CSV or Parquet input with columns `t_ps`, `p_GeV`, `flavour_tag` (or convertible)
- `--remote_url <url>`: HTTP(S) URL to download; supports CSV/Parquet/ROOT
- `--root <path>`: local ROOT file path
- `--preset lhcb_b2hhh`: convenience loader for LHCb B2HHH files (derives p and t)
- `--derive_t`: derive `t_ps` from flight distance (needs `--map_L` and `p_GeV`)
- `--map_L`: flight distance branch/column name when deriving t
- `--map_charge_cols`: comma-separated charge columns used to infer flavour tag
- `--infer_tag_from_charges`: try to infer `flavour_tag` from charge columns (simple sum rule)
- `--allow_untagged`: proceed even if `flavour_tag` is missing (sets tag=0)
- `--p_max` / `--p_clip_quantile`: clip extreme `p_GeV` outliers
- `--n_gbins`, `--dm_hint`, `--Gamma_hint`, `--fix_dm`, `--fix_Gamma`: analysis parameters forwarded to `main.run_analysis`

Notes on tag/untagged data
---------------------------
- The oscillation model used in `main.py` expects a flavour tag `q` per event. If `q==0` for all
events the oscillatory cosine term disappears and the extended model's λ parameter becomes
numerically degenerate — this can cause λ̂ to peg at bounds and weird likelihood differences.
- Use `--infer_tag_from_charges` (and `--map_charge_cols`) to attempt a simple tag inference
	(sum charges → +1/-1/0). If you don't have a tag and still want to run, pass `--allow_untagged`.
	Consider instead adapting the analysis to an untagged likelihood (we can add that if desired).

Troubleshooting
---------------
- If you see λ̂ stuck at bounds and extreme 2ΔNLL values, check:
	1. Are `flavour_tag` values present and non-zero? If not, provide a mapping or use charge-based inference.
	2. Are derived `t_ps` or `p_GeV` sensible? Use the diagnostics printed by the wrapper (I provide sample scripts in `README` to inspect distributions).
	3. Try a smaller subsample (`head` or `sample`) to iterate quickly.

Suggested workflow for LHCb ROOT files
-------------------------------------
1. Download the ROOT file locally (tooling in `gamma_lambda_fit.py` does this if `--remote_url` is used).
2. Run a diagnostic sample to inspect `B_FlightDistance`, daughter momentum components, and whether charge branches exist:

```bash
python - <<'PY'
import uproot, pandas as pd
f = uproot.open('/tmp/B2HHH_MagnetDown.root')
 t = f['DecayTree']
 arr = t.arrays(['B_FlightDistance','H1_PX','H1_PY','H1_PZ','H1_Charge'], library='np')
 df = pd.DataFrame({k: arr[k] for k in arr})
 print(df[['B_FlightDistance','H1_PX','H1_PZ','H1_Charge']].describe())
PY
```

3. If charges are available, run with `--infer_tag_from_charges`.

Examples
--------
- Demo run (synthetic):

```bash
python gamma_lambda_fit.py --demo --out out_demo
```

- Remote LHCb ROOT preset (derives p and t automatically):

```bash
python gamma_lambda_fit.py \
	--remote_url http://opendata.cern.ch/.../B2HHH_MagnetDown.root \
	--preset lhcb_b2hhh --infer_tag_from_charges --out out_remote_root
```

Next steps and contributions
----------------------------
- Add an explicit untagged-likelihood branch in `main.py` so we can fit untagged samples correctly.
- Improve tag inference (machine-learned or experiment-specific tagging algorithms).
- Add unit tests and a `requirements.txt` file for reproducibility.

Contact / author
----------------
This project was modified in-session to add remote loaders and convenience flags for LHCb ROOT files.
If you want me to implement the untagged-likelihood path or run the full re-fit with inferred tags, tell me and I'll implement that next.