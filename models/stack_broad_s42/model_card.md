# Model card — stack_broad_s42

**Variante treinada no corpus broad. Comparação, não servida.**

- Config: stack_tfidf3  |  Política: broad  |  Semente: 42  |  git: f854e84
- Corpus de treino: `corpus_broad.parquet` (versão v1; não é o strict v2)
- sha256 de `model.joblib`: `252bde2bdbea7a972f4ebdd074bae51ef5afee695169f844d87955c791850b9b`
- Limiar: 0.4567  |  macro-F1 teste: 0.7561  |  ROC-AUC teste: 0.8396

Mesma arquitetura do stack strict, política de rótulos broad.

## Uso e limites
Classificador de pesquisa para discurso de ódio em redes sociais EN/PT. Probabilístico; não é oráculo de moderação. Os pesos são research-only, pois o treino inclui bases de licença restrita (HateBR, ToLD-BR): ver `LICENSE-MODEL.md` e `methodology/data_provenance.md`. Limites e viés: `methodology/limitations.md`.
