# Índice de experimentos

Uma linha por run relevante, do mais recente para o mais antigo. A fonte de verdade das
métricas é `reports/metrics/*.json` e o MLflow (`logs/mlruns`). Aqui fica só o resumo
navegável: run_id, config, política de rótulo, modelo, macro-F1 (val/test), observação.

Baselines clássicos (Fase 7). Corpus: strict 32.899 / broad 33.640 linhas; splits
congelados 70/15/15; TF-IDF word(1-2)+char(3-5); limiar ajustado na validação; seed 42.
Fonte de verdade: reports/tables/leaderboard.csv e reports/metrics/*.json.

| modelo | política | test macro-F1 | IC95 | recall ódio | ROC-AUC |
|--------|----------|:-------------:|------|:-----------:|:-------:|
| tfidf_logreg | strict | **0,717** | [0,699, 0,735] | 0,583 | 0,857 |
| tfidf_lgbm | broad | 0,711 | [0,697, 0,725] | 0,544 | 0,782 |
| tfidf_logreg | broad | 0,708 | [0,694, 0,722] | 0,524 | 0,789 |
| tfidf_lgbm | strict | 0,699 | [0,680, 0,719] | 0,468 | 0,843 |
| tfidf_svm | strict | 0,689 | [0,672, 0,707] | 0,504 | 0,808 |
| tfidf_svm | broad | 0,682 | [0,668, 0,696] | 0,493 | 0,758 |

**Leitura.** Regressão Logística é o melhor clássico (strict 0,717). SVM o mais fraco.
strict tem ROC-AUC mais alto (rótulos mais limpos). Por fonte (teste): memes ~0,49 com
recall de ódio ~0 (não-aprendível só com texto), PT 0,64-0,69, tweets EN 0,65-0,76.
LightGBM sobre TF-IDF esparso é lento (~130s/run) e não supera o linear; migrar GBM para
embeddings densos (SBERT) na próxima iteração.

Próximo: adicionar trilha SBERT+classificador e os transformers no Colab (Fase 8), depois
significância pareada (McNemar) e calibração (Fase 9).
