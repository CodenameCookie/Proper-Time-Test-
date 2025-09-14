CODE_EXPLAINER
===============

Purpose
-------
This document provides a plain-text, SEO-friendly explanation of the analysis code in this
repository. It mixes short code excerpts, algebraic descriptions, and high-level steps so search
engines and readers can quickly understand what the project does.

Short summary
-------------
We evaluate whether an extra exponential damping term (λ) improves fits to time-dependent
oscillations of B-hadron decays. The time-dependent PDF used is:

  f(t) = exp(-(Γ + λ) t) * (1 + q A0 cos(Δm t)) / norm

where:
- t is proper time (ps)
- Γ is the baseline decay rate (ps^-1)
- λ is an additional damping parameter (ps^-1)
- q ∈ {+1,-1,0} is the flavour tag
- A0 is the oscillation amplitude
- Δm is the oscillation frequency (ps^-1)

We fit both a baseline model (λ=0) and an extended model (λ free) in different bins of boost
γ = E/m = sqrt(p^2 + m^2)/m to test if λ shows a dependence on γ.

Data flow and core functions
---------------------------
- `load_dataframe(path)` — load CSV or Parquet into a pandas DataFrame.
- `load_cern_root_lhcb_b2hhh(path, ...)` — specialized loader for LHCb B2HHH files; computes
  `p_GeV` from daughter momentum components and derives `t_ps` from `B_FlightDistance`.
- `relativistic_gamma(p_GeV, mass_GeV)` — computes γ = sqrt(p^2 + m^2)/m.
- `oscillation_pdf(t, q, A0, dm, Gamma, lam_extra)` — returns the normalized PDF value for each t.
- `neg_loglike(params, t, q, sigma, model, fixed)` — negative log-likelihood for either baseline or extended.
- `fit_bin_mle(...)` — performs an L-BFGS-B optimization to find MLE parameters per bin.
- `run_analysis(df, out_dir, mass_GeV, n_gbins, dm_hint, Gamma_hint, fix_dm, fix_Gamma)` — bins events by
  γ, fits both models per bin, computes 2ΔNLL and performs a weighted linear trend test on λ̂ vs γ.

Algebraic notes
---------------
- The PDF integrand before normalization is:

  g(t) = exp(-(Γ + λ) t) * (1 + q A0 cos(Δm t))

- The normalization for t ≥ 0 is:

  Z = ∫_0^∞ exp(-(Γ + λ) t) dt + q A0 ∫_0^∞ exp(-(Γ + λ) t) cos(Δm t) dt

  which evaluates to:

  Z = 1/(Γ + λ) + q A0 * (Γ + λ) / ((Γ + λ)^2 + Δm^2)

- Therefore the normalized PDF is f(t) = g(t) / Z. The code ensures numerical stability by clipping values.

Fitting and statistics
----------------------
- For each γ bin we fit:
  - baseline: free parameters (A0, maybe Δm, maybe Γ), λ=0
  - extended: same as baseline but with λ free (bounded between 0 and 2 ps^-1)
- We compute 2ΔNLL = 2*(NLL_extended − NLL_baseline). If dchi > 0 we estimate a per-bin λ uncertainty
  via: σλ ≈ |λ| / sqrt(dchi). If dchi is near zero we fall back to a conservative uncertainty floor.
- Trend test: fit λ̂ vs (γ − meanγ) weighted by 1/σλ^2 to find slope and its z-score.

Practical notes and pitfalls
----------------------------
- Flavour tags: if `flavour_tag` is missing we provide flags to either infer tags from charge columns
  or proceed as untagged (`q=0`). If q=0 globally the oscillatory term is gone and λ is poorly constrained.
- Unit conversions: LHCb files often store momentum in MeV. The loader divides by 1000 to produce GeV.
- Momentum outliers: we added optional clipping flags (`--p_max`, `--p_clip_quantile`) to avoid extreme
  tails influencing γ binning.

Example command lines
---------------------
- Demo run:

  python gamma_lambda_fit.py --demo --out out_demo

- Remote LHCb file using preset and charge-inference:

  python gamma_lambda_fit.py \
    --remote_url http://.../B2HHH_MagnetDown.root \
    --preset lhcb_b2hhh --infer_tag_from_charges --out out_remote_root

SEO tips
--------
- This file contains plaintext math and code snippets so search engines can index the key terms: "B-hadron", "proper time", "oscillation PDF", "lambda damping", "relativistic gamma", "LHCb".

Contact
-------
If you'd like this converted to an HTML explainer or a blog-ready Markdown with figures and rendered formulas, tell me and I will produce it.
