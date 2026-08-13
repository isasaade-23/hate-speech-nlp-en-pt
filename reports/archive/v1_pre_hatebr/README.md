# Arquivo v1 (corpus pré-HateBR, 4 fontes)

Resultados congelados do corpus **v1** (memotion, tweets_ip, pt_fortuna, multioff),
antes da adição do HateBR em 2026-08-11. Substituídos em `reports/` pelos resultados
do corpus **v2** (5 fontes).

- `metrics/` — 24 JSONs de métricas por modelo (clássicos treinados localmente +
  neurais treinados no Colab e mergeados via `notebooks/merge_neural_results.py`)
- `predictions/` — 52 parquets de predições por exemplo (val/test), base do McNemar
- `registry_v1.json` — registry completo do v1 (model_id → config, threshold, métricas)
- `tables/` — tabelas geradas sobre o v1 e ainda não re-rodadas no v2: transfer
  cross-domain/cross-lingual, análise de erro, seleção de produto (Pareto) e
  composição do corpus. Se algum desses experimentos entrar no artigo, re-rodar
  sobre o corpus v2 primeiro.

Os corpora e splits v1 não estão aqui (regeneráveis pelo pipeline a partir dos dados
brutos; ver `data/processed/_pre_hatebr_v1/` local). Os resultados neurais NÃO são
regeneráveis sem GPU, por isso são versionados.

Melhores v1 (test macro-F1): strict xlmr_multilingual 0.7497, tfidf_logreg 0.7094;
broad bertweet_en 0.7481, tfidf_lgbm 0.6983.
