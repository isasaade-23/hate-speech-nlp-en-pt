# Reproducibility

## Environment

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
```

## Local pipeline (classical track, CPU)

```bash
python scripts/download_data.py  # public sources -> data/source_zips (see its output for manual ones)
hsc ingest                       # 8 datasets -> common schema
hsc probe-dataset2               # resolve the unlabeled 1/2/3 tweets source
hsc harmonize --policy strict    # (and --policy broad)
hsc split --policy strict        # leakage-safe frozen split
hsc langid  --policy strict      # language detection
hsc train -c configs/classical/tfidf_logreg.yaml --policy strict
hsc report                       # leaderboard + breakdown + seed-aggregate tables
```

## Deeper evaluation

```bash
hsc analyze      # paired McNemar + calibration
hsc transfer     # cross-lingual / cross-domain
hsc bias         # identity-term bias probe
hsc errors       # qualitative error analysis
hsc product      # product model selection (Pareto)
```

## Transformers (Colab GPU)

Open `notebooks/colab_neural.ipynb`. It is a self-contained notebook that uploads the frozen
corpus and trains XLM-R / BERTimbau / BERTweet under the same protocol (same threshold
tuning, same metric schema). For confidence intervals, set `SEEDS = [42, 43, 44]`. Download
`hsc_neural_results.zip`, then:

```bash
python notebooks/merge_neural_results.py hsc_neural_results.zip
hsc report && hsc analyze && hsc bias
```

## Repository layout

```
configs/     experiment configs (the methodology, as code)
src/hsc/     installable package (ingest ... train ... evaluate ... product ... inference)
api/ demo/ deploy/    FastAPI service, Gradio demo, Docker
methodology/ decision log, provenance, label mapping, experiment log, limitations
tests/       leakage gate + metric + contract tests
assets/      committed figures (regenerate with assets/make_figures.py)
```

Runs are seeded and config-driven; `models/registry.json` links each model id to its
config, git SHA, and metrics.

## Order of the analysis scripts (`scripts/`)

Run from the repo root after the pipeline above (frozen split and `models/registry.json`
in place). Order inferred from each script's docstring and inputs; the scripts are not chained
by a runner. Scripts marked (API) need keys kept outside the repo.

1. **Product model**: `tune_tfidf_logreg.py` (Optuna search), `fit_platt_demo.py` and
   `calibrate_demo.py` (calibration, fit on validation only), `build_stack.py` (served stack).
2. **Comparison on the v5 test**: `adversarial_slice.py`, `leaderboard_v5.py`.
3. **Phase 0, PT audit**: `auditoria_pt.py` (integrity, learning curve), `llm_label.py` (API,
   judge of the label), `hatebr_partidarismo.py` (API, partyism under strict v2).
4. **PT recall and identity bias**: `pt_gap.py`, `pt_boost.py`, `pt_identity_probe.py`,
   `pt_threshold_bias.py`, `pt_ensemble.py`, then the release gate `bias_gate_v5.py`.
5. **Step 2.2**: `commercial_retrain_compare.py` (retrain on the commercial corpus).
6. **Step 2.3**: `perturbation_robustness.py` (obfuscation, identity terms).
7. **Step 2.4**: `translate_pilot.py` (MarianMT and Qwen), `translate_gemini.py` (API),
   `avaliar_traducao.py` (quality of the 500 pilot translations).
8. **Step 2.5**: `generate_synthetic.py` (PT-BR synthetic data, local Ollama, seeds from
   permissive sources only).
9. **Extension and field tests**: `export_extension_model.py`, `field_harvest.py`,
   `field_score.py`.
10. **Side probe**: `probe_lfm25_encoder.py` (frozen LFM2.5 embeddings, CPU head).

## Releasing and DOI (Zenodo)

The repository ships a `.zenodo.json` with citation metadata. To mint a DOI:

1. Sign in to [Zenodo](https://zenodo.org) with GitHub and flip the repository on under
   *Settings -> GitHub*.
2. On GitHub, create a release (for example tag `v1.0.0`). Zenodo archives that release and
   issues a DOI automatically.
3. Add the DOI badge Zenodo provides to the README.

The documentation site is built and published by `.github/workflows/docs.yml` on every push
to `main` (enable GitHub Pages with the `gh-pages` branch as source).
