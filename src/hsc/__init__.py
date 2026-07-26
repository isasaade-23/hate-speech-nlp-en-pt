"""hsc — bilingual (EN/PT) hate-speech classifier.

Package layout:
    ingest/     one loader per source dataset -> common schema (data/interim)
    harmonize   apply configs/labels.yaml, build unified corpus (data/processed)
    clean       text normalization (light + heavy profiles)
    langid      language-detection front-end
    splits      leakage-safe stratified group splitting
    features/   tfidf, embeddings
    models/     classical, transformer, bilstm
    train/eval  config-driven training and evaluation
"""

__version__ = "0.1.0"
