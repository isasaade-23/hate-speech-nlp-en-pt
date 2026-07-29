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

## Fase 9: Avaliação aprofundada (2026-07-27)

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
tfidf_logreg strict 0,709 / tfidf_lgbm broad 0,698); fica ~1 a 2 pontos atrás em macro-F1,
com recall de ódio comparável. Sobre embeddings densos o LightGBM finalmente iguala o
linear (habitat certo) e roda em segundos (vs ~130s no TF-IDF esparso). A vantagem real do
SBERT aparece na transferência cross-lingual (abaixo), não in-distribution.

### Significância pareada (McNemar exato, Holm α=0,05): reports/tables/mcnemar_test.csv
- Topo: em **strict tfidf_logreg > tfidf_lgbm** é significativo (p≈0); em **broad
  tfidf_lgbm ≈ tfidf_logreg** empatam (p=1,0). O melhor clássico depende da política.
- 6/10 pares significativos em strict, 2/10 em broad (rótulos broad mais ruidosos).
- SVM segue o mais fraco.

### Calibração (ECE/MCE/Brier): reports/tables/calibration_test.csv + figuras
- **sbert_lgbm é dos mais calibrados** (strict ECE 0,032; broad 0,056), bem melhor que
  tfidf_logreg/lgbm (strict ECE ~0,16; broad ~0,09-0,10). tfidf_svm tem ECE baixo por ser
  conservador (mas MCE alto).
- sbert_logreg strict é o pior (ECE 0,284). Probabilidades infladas.
- Implicação de produto: se o score precisa ser interpretável, sbert_lgbm é preferível
  apesar de ~2 pontos a menos de macro-F1.

### Transferência cross-domínio e cross-lingual: reports/tables/transfer_{strict,broad}.csv

Resultado-título (política broad, densidades de rótulo comparáveis entre fontes):

| experimento | TF-IDF macro-F1 (recall ódio) | SBERT macro-F1 (recall ódio) |
|-------------|:-----------------------------:|:----------------------------:|
| EN→PT (zero-shot) | 0,418 (0,012) | **0,626 (0,373)** |
| PT→EN (zero-shot) | 0,445 (0,075) | **0,674 (0,664)** |
| tweets→memes (cross-domínio EN) | 0,504 (0,148) | **0,540 (0,308)** |
| memes→tweets (cross-domínio EN) | 0,501 (0,212) | **0,550 (0,306)** |

**Leitura.** Cross-lingual: TF-IDF de palavra COLAPSA (recall de ódio ~0,01 a 0,08; não
detecta ódio no outro idioma, por não compartilhar vocabulário); o SBERT transfere de fato
(0,63-0,67; recall 0,37-0,66). Cross-domínio EN o SBERT também ganha, mas fica ~0,55 (meme
segue difícil só-texto). Esta é a Fig. 5 e a medida direta do risco de domain/language
shift. Nota: a correção do UTF-8 elevou o SBERT EN→PT de 0,583 para 0,626 (o tokenizer
multilíngue estava engasgando no PT corrompido); confirma o impacto do bug de encoding.

### Análise de erro qualitativa: reports/tables/error_{modes,rates,examples}_*.csv/md
Melhor modelo strict (tfidf_logreg): 568 erros (313 FN / 255 FP).
- **206 dos 313 FN são "ódio implícito"** (sem token de palavrão). O modelo perde ódio
  sutil/sem slur. É a maior fatia dos falso-negativos.
- **70 FP são "over-flag por slur"** (não-ódio com palavrão marcado como ódio). O modelo
  super-confia na presença de palavrão.
- **PT tem o maior erro (0,277), puxado por falso-positivo (0,185 vs 0,027 no EN)**: o
  modelo super-marca PT como ódio (fonte menor, fronteira ofensa/ódio mais densa).
- Erros por texto curto (7) e por divergência de langid (7) são raros.
- FN "confiantes" em memes são majoritariamente memes cujo ódio está na IMAGEM, não no
  OCR (texto benigno rotulado como ódio). Reforça a limitação só-texto.

### Sondagem de viés por termo de identidade: reports/tables/bias_identity_fpr_*.csv
FPR em linhas NÃO-ódio que citam um grupo (termos neutros), vs. FPR de fundo:
- **orientação sexual é o grupo mais super-marcado**: broad tfidf_lgbm FPR 0,754 vs fundo
  0,165 (gap **+0,589**); citar "gay/lésbica/trans" em texto benigno dispara ódio ~75%.
- religião, nacionalidade/imigração e gênero também mostram gaps grandes (+0,25 a +0,41).
- **SBERT tende a gaps menores que TF-IDF** (menos gatilhado por termo isolado), mas ainda
  positivos. Viés não-intencional clássico (Dixon et al.). Entra no Ethics statement.

---

## Fase 8: Resultados neurais + comparação final clássico vs neural (2026-07-27)

Transformers treinados no Colab (T4, seed 42, fp16), MESMO corpus/split/limiar dos
clássicos. Fonte de verdade: reports/tables/leaderboard.csv (16 modelos).

| modelo | política | test macro-F1 | recall ódio | ROC-AUC |
|--------|----------|:-------------:|:-----------:|:-------:|
| xlm-roberta-base | strict | **0,750** | 0,554 | 0,855 |
| bertimbau (PT) | strict | 0,747 | **0,796** | 0,853 |
| bertweet (EN) | broad | 0,748 | 0,613 | 0,826 |
| xlm-roberta-base | broad | 0,732 | 0,585 | 0,809 |
| bertimbau (PT) | broad | 0,712 | 0,561 | 0,834 |
| bertweet (EN) | strict | 0,707 | 0,518 | 0,817 |

**Resultado central (a tese do artigo).** Os transformers superam o melhor clássico nas
duas políticas: strict 0,750 (XLM-R) vs 0,709 (tfidf_logreg); broad 0,748 (BERTweet) vs
0,698 (tfidf_lgbm). Cerca de +0,04 macro-F1. **McNemar + Holm confirma significância**:
XLM-R > tfidf_logreg strict (p=0,0026); BERTweet > tfidf_lgbm broad (p≈0).

**Calibração (nuance para o produto).** Os neurais NÃO calibram melhor: sbert_lgbm e
tfidf_svm seguem os mais calibrados (ECE strict 0,03), enquanto XLM-R/BERTweet têm MCE
alto. Vitória neural em F1/significância, mas não em calibração → decisão de produto por
Pareto, não só por F1.

**Viés (transformers ajudam).** No over-flag de identidade, os transformers de verdade têm
os menores gaps: XLM-R/BERTweet ~0,258 em orientação sexual (broad) vs TF-IDF 0,42-0,59;
BERTimbau 0,35; SBERT no meio. Ainda positivo em todos. O viés persiste.

**Análise de erro (XLM-R strict vs tfidf_logreg strict).** O transformer erra menos
(506 vs 568), com menos falso-negativo (260 vs 313) e **menos ódio implícito perdido
(183 vs 206)**, coerente com um modelo contextual pegar ódio sem slur. Custo: super-marca
por slur um pouco mais (88 vs 70 FP).

**Ganho do fix de encoding.** BERTimbau PT strict com recall de ódio 0,796 sobre texto PT
correto (UTF-8). O modelo dedicado ao PT rende quando os acentos não estão corrompidos.

---

## Fase 11: Produto + release (2026-07-27)

Seleção do modelo de produto por Pareto (`hsc product`; detalhe em product_decision.md).
Eixos medidos: macro-F1, recall de ódio, ECE, viés de identidade, latência p50/p95 (CPU),
tamanho, licença. Decisão por perfil:
- **melhor qualidade:** XLM-R (GPU, research-only);
- **MVP CPU (recomendado):** tfidf_logreg. p95 1,6 ms, 3,6 MB, autossuficiente, F1 a ~4 pts;
- **score calibrado:** sbert_lgbm (melhor ECE, mas +470 MB de encoder).
Licença: todos research-only (corpus completo); comercial exige retreino só na whitelist.

Dois fixes de deploy no caminho: (1) `EmbeddingVectorizer.__getstate__` não pickla o
encoder → sbert_logreg_strict 479 MB → 3 KB; (2) `inference.py` serve o melhor joblib
LOCAL (neurais ficam no Colab), default = tfidf_logreg_strict.

**Validação da API (uvicorn local, sem Docker).** `/health` OK; `/predict` retorna schema
completo (label, score, idioma, latência ~35 ms). Confirmado EN + PT COM ACENTO (via
UTF-8 correto): "você é incrível" → not_hate 0,145; "seus imbecis..." → hate 0,665;
"parabéns..." → not_hate 0,277. Produto funciona ponta a ponta.

**Release.** Repo público em github.com/isasaade-23/hate-speech-nlp-en-pt (11 commits):
README em inglês sem emojis, capa SVG na IDV, figuras-herói (assets/), CITATION.cff, topics.

**Fase 12 (Docker):** arquivos prontos e revisados (Dockerfile CPU-slim non-root +
healthcheck, requirements-serve enxuto, .dockerignore com !models). Build BLOQUEADO por
WSL2 desativado no Windows 11 Home (erro Wsl/0x80070422); habilitar features
Microsoft-Windows-Subsystem-Linux + VirtualMachinePlatform (admin) e reiniciar. Depois:
`docker compose -f deploy/docker-compose.yml up --build`.

Próximo: build Docker pós-reboot; opcional multi-seed neural (42/43/44) para IC; Fase 13
(mkdocs + DOI Zenodo).

---

## Ablação: remoção de stop words (EN+PT) no TF-IDF (28/07)

**Hipótese testada.** Remover palavras funcionais (preposições, pronomes, artigos e
conjunções, EN+PT; **negações preservadas** pra não inverter sentido) melhoraria os modelos
TF-IDF. Aplicado só ao analisador **word** (o char_wb fica intacto). Lista bilíngue curada
em `src/hsc/features/stopwords.py`; toggle `stop_words: enpt` nos configs `*_nostop.yaml`.
Mesmo split congelado, seed 42.

**Resultado (test set; baseline → sem-stop).** Lista corrigida: **"no" omitido** (é preposição
PT mas também a negação EN. Removê-la inverteria sentido em inglês).

| Modelo | Política | F1 base→sem | ΔF1 | AUC base→sem | ΔAUC | recall ódio base→sem |
|--------|----------|-------------|--------|--------------|--------|----------------------|
| logreg | strict   | 0.709→0.714 | +0.005 | 0.841→0.840 | −0.001 | 0.463→0.544 |
| logreg | broad    | 0.698→0.694 | −0.004 | 0.768→0.768 | −0.001 | 0.548→0.546 |
| lgbm   | strict   | 0.707→0.701 | −0.005 | 0.820→0.818 | −0.002 | 0.558→0.540 |
| lgbm   | broad    | 0.698→0.698 | −0.001 | 0.768→0.769 | +0.001 | 0.551→0.521 |
| svm    | strict   | 0.672→0.664 | −0.007 | 0.772→0.773 | +0.001 | 0.497→0.521 |
| svm    | broad    | 0.673→0.665 | −0.009 | 0.736→0.734 | −0.002 | 0.525→0.468 |

**Conclusão (negativa).** Nenhum ganho de macro-F1: ΔF1 de −0.009 a +0.005 (média ≈ −0.4 pt),
dentro do IC 95%. **A AUC (qualidade de ranqueamento, independente de limiar) é praticamente
idêntica** (ΔAUC −0.002 a +0.001, média ≈ −0.06 pt): a capacidade discriminativa não muda. O
recall de ódio fica **misto** (3 sobem, 3 caem). Nem o "mais recall" se sustenta.

**Interpretação.** Char n-gram (char_wb 3-5) + IDF já absorvem as palavras funcionais (IDF pesa
pouco termo frequente), então remover pronomes/preposições é **redundante e por vezes levemente
prejudicial**. Descarta dêixis útil ("those/vocês"). **Decisão: manter o pré-processamento atual**;
modelo de produto inalterado. SBERT/transformers não testados de propósito (representação contextual
da frase inteira; remoção quebraria a estrutura). Viz interativa (F1/AUC/recall toggle) gerada como
artifact; toggle no código via `stop_words: enpt`.

## Diagnóstico: por que a LogReg é competitiva (29/07)

**Pergunta.** O melhor clássico (tfidf_logreg, 0.709 strict) fica a ~4 pontos do melhor
transformer. Por que um modelo linear simples chega tão perto, e por que ele bate os outros
clássicos?

**Evidência 1: os pesos são léxico explícito e bilíngue.** Os 20 n-gramas de palavra de maior
peso positivo do tfidf_logreg strict são palavrão e ataque de identidade em EN e PT (`islam`,
`hindus`, `traitors`, `kill`, `gorda`, `burra`, mais os slurs). 19 dos 50 maiores pesos gerais
são char n-gramas (char_wb 3-5), que capturam variação e ortografia. Detecção de ódio neste
corpus é, em boa parte, um problema de léxico, e TF-IDF word+char sobre modelo linear resolve
isso perto do teto.

**Evidência 2: McNemar dentro da família clássica** (reports/tables/mcnemar_test.csv).
- strict: logreg supera lgbm com significância (221 vs 126 discordantes, p≈0, Holm). Não é empate.
- broad: logreg e lgbm empatam exato (290 vs 290, p=1).
- SVM é pior nas duas por **ranqueamento**, não só por limiar: AUC 0.772 vs 0.841 do logreg. A
  hinge loss produz score que ordena pior neste espaço esparso.
- SBERT congelado perde os tokens de superfície, então fica abaixo do TF-IDF dentro do domínio
  (sbert_lgbm strict 0.683 vs tfidf_logreg 0.709) mas transfere entre línguas (ver Fase 9).

**Conclusão.** O resultado não é anômalo. Em espaço esparso de alta dimensão (84.737 features),
quase linearmente separável, o modelo linear é quase ótimo. Árvore (LGBM) fragmenta o sinal e no
máximo empata; SVM ordena pior. A vantagem do transformer não vem dos casos explícitos, que o
linear já acerta, e sim de contexto (ódio implícito) e transferência cross-lingual.

## Ensemble superfície + semântica (29/07)

**Método.** Combinar tfidf_logreg (superfície lexical) com o melhor transformer por política
(semântica), por média ponderada e por stacking (regressão logística sobre os scores). Peso e
limiar afinados no val; avaliação no test. Sem vazamento: o test só é tocado no fim. Usa as
predições por exemplo já persistidas em reports/predictions/.

| Política | melhor single | melhor ensemble (F1) | ΔF1 | recall ódio (single → ensemble ponderado) |
|----------|---------------|----------------------|-----|-------------------------------------------|
| strict   | xlmr 0.749    | stack 0.748          | −0.001 | 0.554 → 0.626 |
| broad    | bertweet 0.750| stack+sbert 0.754    | +0.004 | 0.607 → 0.653 |

**Conclusão.** No macro-F1 o ensemble empata (os transformers já estão no teto do dado). O valor
está em outro lugar: a média ponderada sobe o **recall de ódio em 5 a 9 pontos** e melhora a AUC
nas duas políticas, ao custo de fração de ponto de F1. É a métrica eticamente crítica. Entra no
artigo como resultado "surface + semantic": o clássico pega o ódio explícito por limiar, o
transformer pega o implícito, e a combinação recupera parte dos falsos-negativos implícitos
(206 dos 313 FN eram ódio implícito, ver Fase 9).

## TabPFN (foundation model tabular) sobre features densas (29/07)

**Setup.** TabPFN v2 (pacote tabpfn 8.2) sobre duas bases densas de features, **amostra toda na
GPU** (Colab, notebooks/colab_tabpfn.ipynb): embeddings SBERT (384 dim) e TF-IDF word+char
reduzido por TruncatedSVD para 300 dim. Comparado, nas MESMAS features, com LightGBM e LogReg.
Limiar afinado no val, avaliação no test.

| Features | TabPFN (strict / broad) | LightGBM | LogReg |
|----------|-------------------------|----------|--------|
| SBERT        | **0.684 / 0.699** | 0.675 / 0.681 | 0.638 / 0.685 |
| TF-IDF→SVD300 | **0.676 / 0.691** | 0.671 / 0.664 | 0.670 / 0.686 |

Referência: TF-IDF esparso completo 0.709 / 0.698; transformers 0.750.

**McNemar, TabPFN(SBERT) vs melhor clássico.** strict vs tfidf_logreg: 225 vs 253 discordantes,
p=0.22 (empate). broad vs tfidf_lgbm: 470 vs 483, p=0.70 (empate).

**Conclusão.** Duas coisas ao mesmo tempo. O TabPFN é o **melhor classificador sobre features
densas** (bate LGBM e LogReg nas duas bases e nas duas políticas). E ele **empata
estatisticamente com o melhor clássico**, sem passar do TF-IDF esparso nem chegar aos
transformers. O TF-IDF→SVD não ajudou o TabPFN a superar o SBERT: a compressão densa (SBERT ou
SVD) descarta os tokens raros de superfície que carregam o sinal. Reforça a tese central: o
gargalo é a **representação**, não o classificador. Um run anterior em CPU com treino subamostrado
(n=2000) deu 0.621 e enganou; a amostra toda na GPU corrige. Predições em
reports/predictions/tabpfn_*. Não incluído no leaderboard.csv (gerado por `hsc report` a partir
dos runs dirigidos por config); registrado aqui como experimento exploratório.
