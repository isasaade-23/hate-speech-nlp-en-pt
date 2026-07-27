# Reproducibility

## Environment

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
```

## Local pipeline (classical track, CPU)

```bash
hsc ingest                       # 4 datasets -> common schema
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

Open `notebooks/colab_neural.ipynb` — a self-contained notebook that uploads the frozen
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

## Releasing and DOI (Zenodo)

The repository ships a `.zenodo.json` with citation metadata. To mint a DOI:

1. Sign in to [Zenodo](https://zenodo.org) with GitHub and flip the repository on under
   *Settings -> GitHub*.
2. On GitHub, create a release (for example tag `v1.0.0`). Zenodo archives that release and
   issues a DOI automatically.
3. Add the DOI badge Zenodo provides to the README.

The documentation site is built and published by `.github/workflows/docs.yml` on every push
to `main` (enable GitHub Pages with the `gh-pages` branch as source).
