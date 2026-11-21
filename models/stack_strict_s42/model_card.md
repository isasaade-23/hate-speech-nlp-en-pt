# Model card — stack_strict_s42

**MODELO SERVIDO (API, demo, base do modelo da extensão).**

- Config: stack_tfidf3  |  Política: strict  |  Semente: 42  |  git: c182003
- Corpus de treino: `corpus_strict.parquet` (versão v1; não é o strict v2)
- sha256 de `model.joblib`: `a17ad071446e7dbdc873983be8abcf94249b1fa16b9fb7ec31debc9b32294753`
- Limiar: 0.3372  |  macro-F1 teste: 0.7906  |  ROC-AUC teste: 0.8875

Stack TF-IDF (3 vetorizadores) com logreg + svm + lightgbm. Limiar único; o mapa por idioma está vazio de propósito (ver `threshold_policy` no registry e `methodology/pt_recall_x_vies.md`). Recall de ódio em PT: 0,3162 com este limiar.

## Uso e limites
Classificador de pesquisa para discurso de ódio em redes sociais EN/PT. Probabilístico; não é oráculo de moderação. Os pesos são research-only, pois o treino inclui bases de licença restrita (HateBR, ToLD-BR): ver `LICENSE-MODEL.md` e `methodology/data_provenance.md`. Limites e viés: `methodology/limitations.md`.
