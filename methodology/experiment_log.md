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

---

## Fase 9 — Avaliação aprofundada (2026-07-27)

Fonte de verdade: reports/tables/{leaderboard,mcnemar_test,calibration_test,transfer_*}.csv
e reports/figures/calibration_*.png. Seed 42; mesmos splits congelados.

### Trilha SBERT (paraphrase-multilingual-MiniLM-L12-v2, encoder congelado, 384 dim)

| modelo | política | test macro-F1 | recall ódio | ROC-AUC |
|--------|----------|:-------------:|:-----------:|:-------:|
| sbert_lgbm | broad | 0,695 | **0,672** | 0,773 |
| sbert_logreg | broad | 0,692 | 0,601 | 0,768 |
| sbert_lgbm | strict | 0,689 | 0,503 | 0,814 |
| sbert_logreg | strict | 0,669 | 0,556 | 0,808 |

**Leitura.** In-distribution o SBERT congelado NÃO supera o TF-IDF (melhor clássico segue
tfidf_logreg strict 0,717 / tfidf_lgbm broad 0,711). Mas o SBERT entrega **recall de ódio
bem mais alto** (sbert_lgbm broad 0,672 vs tfidf_lgbm broad 0,544) — troca relevante
eticamente. Sobre embeddings densos o LightGBM finalmente iguala o linear (habitat certo)
e roda em segundos (vs ~130s no TF-IDF esparso). Encoding do corpus é cacheado por texto.

### Significância pareada (McNemar exato, Holm α=0,05) — reports/tables/mcnemar_test.csv
- **tfidf_lgbm vs tfidf_logreg: NÃO significativo** nas duas políticas (empatam no topo).
- **tfidf (logreg/lgbm) > SBERT** significativo em broad; em strict o tfidf_logreg vs
  sbert_lgbm não é significativo (p=0,13).
- SVM é consistentemente o mais fraco (perde significativamente para os GBMs).

### Calibração (ECE/MCE/Brier) — reports/tables/calibration_test.csv + figuras
- **sbert_lgbm é dos mais calibrados** (broad ECE 0,054; strict 0,037), melhor que os
  TF-IDF (logreg/lgbm broad ECE ~0,095; strict ~0,15).
- sbert_logreg strict é o pior calibrado (ECE 0,273) — probabilidades infladas.
- Implicação de produto: se o score precisa ser interpretável, sbert_lgbm é preferível
  apesar de ~2 pontos a menos de macro-F1.

### Transferência cross-domínio e cross-lingual — reports/tables/transfer_{strict,broad}.csv

O resultado-título (política broad, densidades de rótulo comparáveis entre fontes):

| experimento | TF-IDF macro-F1 (recall ódio) | SBERT macro-F1 (recall ódio) |
|-------------|:-----------------------------:|:----------------------------:|
| EN→PT (zero-shot) | 0,419 (0,012) | **0,583 (0,310)** |
| PT→EN (zero-shot) | 0,447 (0,069) | **0,680 (0,583)** |
| tweets→memes (cross-domínio EN) | 0,514 (0,198) | 0,513 (0,214) |
| memes→tweets (cross-domínio EN) | 0,485 (0,158) | **0,554 (0,398)** |

**Leitura.** Cross-lingual: TF-IDF de palavra COLAPSA (recall de ódio ~0,01-0,07 — não
detecta ódio no outro idioma, como esperado por não compartilhar vocabulário); o SBERT
transfere de fato (recall 0,31-0,58). Cross-domínio tweets→memes fica ~0,5 nos dois
(meme não é aprendível só-texto, coerente com o in-distribution). Em strict o sinal
cross-lingual é mais ruidoso (ódio raríssimo nas fontes EN strict), então broad é a
vitrine. Esta é a Fig. 5 do artigo e a medida direta do risco de domain/language shift.

Próximo: transformers no Colab (Fase 8) entram no mesmo leaderboard/McNemar; depois
análise de erro qualitativa e sondagem de viés por termo de identidade.
