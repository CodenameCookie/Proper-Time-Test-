Purpose
-------
This document describes how we test for an extra exponential damping term (λ) in the time-dependent
oscillations of neutral B-hadron decays, and why such a test may reveal new physics or systematic
effects.

1. Objective
------------

We test the model:

    f(t) = e^{-(\Gamma + \lambda) t} (1 + q A_0 cos(\Delta m t)) / Z

where

- t = proper decay time (ps)
- Γ = known decay width (ps⁻¹)
- λ = potential extra damping (ps⁻¹)
- q = flavour tag (+1 for B⁰, −1 for anti-B⁰, 0 if unknown)
- A₀ = oscillation amplitude
- Δm = mass splitting (oscillation frequency)
- Z = normalization constant (integral over t ≥ 0)

We examine whether λ is nonzero, and whether λ depends on relativistic boost γ = sqrt(p^2 + m^2)/m.

2. Motivation
-------------

- A nonzero λ would indicate quantum decoherence, environmental coupling, or physics beyond the
  Standard Model affecting neutral meson mixing.
- Tight constraints on λ strengthen confidence in CP violation and lifetime measurements used across
  particle physics.
- A γ-dependent λ probes energy dependence or Lorentz-violating effects and helps distinguish
  systematic from physical origins.

3. Methodology
--------------

- Data sources: LHCb Open Data (ROOT), local CSV/Parquet, or synthetic `selftest` datasets.
- Per-γ-bin procedure:
  1. Compute γ = sqrt(p^2 + m^2)/m using `p_GeV` and `mass_GeV`.
  2. Fit baseline model (λ = 0) and extended model (λ free) by MLE.
  3. Record λ̂, NLL_baseline, NLL_extended, and compute 2ΔNLL = 2(NLL_extended − NLL_baseline).
  4. Perform a weighted linear trend test of λ̂ vs γ to estimate slope and z-score.

4. Possible outcomes and interpretation
---------------------------------------

Outcome | Interpretation
-------:|:--------------
λ ≈ 0, slope ≈ 0 | No extra damping; sets strong limits on decoherence
λ > 0, slope ≈ 0 | Uniform extra damping; likely experimental/systematic origin
λ > 0, slope > 0 (significant) | Energy-dependent decoherence or new physics candidate

In any case, we must consider systematics: time-resolution modelling, mis-tagging, unit conversion,
background contamination, and selection biases.

5. Broader impacts and applications
----------------------------------

- Signal-processing & time-series analysis: improved resolution convolution and deconvolution
  methods apply to medical imaging, radar/lidar, and seismology.
- Statistical toolchains: robust MLE and model-comparison tools aid domains needing small-signal
  detection (epidemiology, economics).
- Quantum technologies: identifying decoherence mechanisms informs design and mitigation in
  quantum sensors and computing platforms.
- Reproducible data pipelines: portable loaders and clear docs lower barriers for open-data science
  in education and industry.

6. Validation and robustness checks
-----------------------------------

- Use synthetic `selftest` datasets (null and injected-signal) to validate fitter behavior.
- Vary analysis choices: γ-binning, clipping thresholds, tag-handling, and time-resolution model.
- Check unit conversions (MeV↔GeV), flight-distance units, and verify normalization conventions.
- Quantify sensitivity to mis-tagging and missing flavour-tag information.

7. Next practical steps
-----------------------

1. Implement robust tag inference from charge columns (if present) and rerun on a subsample.
2. Add an untagged-likelihood option to `main.py` to analyze untagged datasets properly.
3. Add unit tests, CI, and a reproducible script to re-run canonical analysis and compare outputs.
4. Prepare a public-friendly summary (blog / grant-style) if we obtain interesting constraints.

References & notes
------------------

If desired, I can add pointers to relevant theoretical papers on decoherence, LHCb dataset identifiers,
and recent experimental limits on damping parameters for neutral meson systems.

Current experimental status / comparison
----------------------------------------
Recent experimental work (Alok et al., Jan 2025; arXiv:2501.03136) reports nonzero decoherence
parameters in combined analyses of neutral B mesons. Key takeaways:

- The analysis reports a nonzero decoherence parameter (\lambda_d) for B_d mesons at ~6σ.
- It provides a first experimental constraint on (\lambda_s) for B_s mesons at ~3σ.
- The paper shows that including decoherence in global fits can shift extracted values of \Delta m
  and CP-asymmetry parameters compared to fits that assume perfect coherence.

How to use this result in our work:

- Benchmark: compare our pipeline sensitivity (per-bin z-scores, λ̂ uncertainties, and slope vs γ)
  with the sensitivity reported in Alok et al. If we cannot match sensitivity, document the sources
  (event count, tagging, or analysis choices).
- Target: inject λ values at the same order of magnitude into synthetic `selftest` datasets and
  verify the pipeline recovers them with comparable significance.
- Add a "Table of prior bounds" if you want numeric values from the paper (I can extract and insert them).

