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

Fonte de verdade: reports/tables/{leaderboard,mcnemar_test,calibration_test,transfer_*,
error_*,bias_*}.csv e reports/figures/calibration_*.png. Seed 42; splits congelados.
**Todos os números abaixo são pós-correção do encoding UTF-8 do PT** (ver
DECISOES_METODOLOGICAS 2026-07-27); substituem a primeira passada sobre PT corrompido.

### Trilha SBERT (paraphrase-multilingual-MiniLM-L12-v2, encoder congelado, 384 dim)

| modelo | política | test macro-F1 | recall ódio | ROC-AUC |
|--------|----------|:-------------:|:-----------:|:-------:|
| sbert_lgbm | broad | 0,690 | 0,529 | 0,770 |
| sbert_logreg | broad | 0,685 | 0,568 | 0,765 |
| sbert_lgbm | strict | 0,683 | 0,520 | 0,808 |
| sbert_logreg | strict | 0,665 | 0,487 | 0,801 |

**Leitura.** In-distribution o SBERT congelado NÃO supera o TF-IDF (melhor clássico:
tfidf_logreg strict 0,709 / tfidf_lgbm broad 0,698); fica ~1-2 pontos atrás em macro-F1,
com recall de ódio comparável. Sobre embeddings densos o LightGBM finalmente iguala o
linear (habitat certo) e roda em segundos (vs ~130s no TF-IDF esparso). A vantagem real do
SBERT aparece na transferência cross-lingual (abaixo), não in-distribution.

### Significância pareada (McNemar exato, Holm α=0,05) — reports/tables/mcnemar_test.csv
- Topo: em **strict tfidf_logreg > tfidf_lgbm** é significativo (p≈0); em **broad
  tfidf_lgbm ≈ tfidf_logreg** empatam (p=1,0). O melhor clássico depende da política.
- 6/10 pares significativos em strict, 2/10 em broad (rótulos broad mais ruidosos).
- SVM segue o mais fraco.

### Calibração (ECE/MCE/Brier) — reports/tables/calibration_test.csv + figuras
- **sbert_lgbm é dos mais calibrados** (strict ECE 0,032; broad 0,056), bem melhor que
  tfidf_logreg/lgbm (strict ECE ~0,16; broad ~0,09-0,10). tfidf_svm tem ECE baixo por ser
  conservador (mas MCE alto).
- sbert_logreg strict é o pior (ECE 0,284) — probabilidades infladas.
- Implicação de produto: se o score precisa ser interpretável, sbert_lgbm é preferível
  apesar de ~2 pontos a menos de macro-F1.

### Transferência cross-domínio e cross-lingual — reports/tables/transfer_{strict,broad}.csv

Resultado-título (política broad, densidades de rótulo comparáveis entre fontes):

| experimento | TF-IDF macro-F1 (recall ódio) | SBERT macro-F1 (recall ódio) |
|-------------|:-----------------------------:|:----------------------------:|
| EN→PT (zero-shot) | 0,418 (0,012) | **0,626 (0,373)** |
| PT→EN (zero-shot) | 0,445 (0,075) | **0,674 (0,664)** |
| tweets→memes (cross-domínio EN) | 0,504 (0,148) | **0,540 (0,308)** |
| memes→tweets (cross-domínio EN) | 0,501 (0,212) | **0,550 (0,306)** |

**Leitura.** Cross-lingual: TF-IDF de palavra COLAPSA (recall de ódio ~0,01-0,08 — não
detecta ódio no outro idioma, por não compartilhar vocabulário); o SBERT transfere de fato
(0,63-0,67; recall 0,37-0,66). Cross-domínio EN o SBERT também ganha, mas fica ~0,55 (meme
segue difícil só-texto). Esta é a Fig. 5 e a medida direta do risco de domain/language
shift. Nota: a correção do UTF-8 elevou o SBERT EN→PT de 0,583 para 0,626 (o tokenizer
multilíngue estava engasgando no PT corrompido); confirma o impacto do bug de encoding.

### Análise de erro qualitativa — reports/tables/error_{modes,rates,examples}_*.csv/md
Melhor modelo strict (tfidf_logreg): 568 erros (313 FN / 255 FP).
- **206 dos 313 FN são "ódio implícito"** (sem token de palavrão) — o modelo perde ódio
  sutil/sem slur. É a maior fatia dos falso-negativos.
- **70 FP são "over-flag por slur"** (não-ódio com palavrão marcado como ódio) — o modelo
  super-confia na presença de palavrão.
- **PT tem o maior erro (0,277), puxado por falso-positivo (0,185 vs 0,027 no EN)**: o
  modelo super-marca PT como ódio (fonte menor, fronteira ofensa/ódio mais densa).
- Erros por texto curto (7) e por divergência de langid (7) são raros.
- FN "confiantes" em memes são majoritariamente memes cujo ódio está na IMAGEM, não no
  OCR (texto benigno rotulado como ódio) — reforça a limitação só-texto.

### Sondagem de viés por termo de identidade — reports/tables/bias_identity_fpr_*.csv
FPR em linhas NÃO-ódio que citam um grupo (termos neutros), vs. FPR de fundo:
- **orientação sexual é o grupo mais super-marcado**: broad tfidf_lgbm FPR 0,754 vs fundo
  0,165 (gap **+0,589**); citar "gay/lésbica/trans" em texto benigno dispara ódio ~75%.
- religião, nacionalidade/imigração e gênero também mostram gaps grandes (+0,25 a +0,41).
- **SBERT tende a gaps menores que TF-IDF** (menos gatilhado por termo isolado), mas ainda
  positivos. Viés não-intencional clássico (Dixon et al.) — entra no Ethics statement.

Próximo: transformers no Colab (Fase 8) entram no mesmo leaderboard/McNemar/calibração;
depois seleção do modelo de produto (Pareto F1×latência×tamanho×licença×viés).
