# Auditoria do pipeline em português (Fase 0, itens 1 a 3)

Gerado por `scripts/auditoria_pt.py`. Corpus `corpus_strict.parquet`, fatia PT (33238 textos: {'train': 23743, 'test': 4748, 'val': 4747}). Só agregados; nenhum texto.

`--juiz` refaz só a seção 4 sobre este arquivo; `--amostra` refaz a amostra do item 4.

## 1. Integridade

**Repetidos exatos** (texto limpo normalizado): 0 grupos com mais de um texto.

**Quase repetidos** (MinHash 5-gramas de caractere, só PT):

| limiar_jaccard | clusters_com_2+ | textos_em_clusters | clusters_entre_splits | test_com_par_no_treino | test_com_par_no_treino_odio | clusters_rotulo_conflitante | clusters_entre_fontes |
|---|---|---|---|---|---|---|---|
| 0.9000 | 120 | 256 | 0 | 0 | 0 | 17 | 2 |
| 0.7000 | 537 | 1279 | 218 | 116 | 23 | 65 | 14 |

O build agrupa por cluster a 0,90 antes do split, então a linha de 0,90 deve dar zero entre splits. A de 0,70 mede o que escapa a esse limiar. Ids em `reports/tables/auditoria_pt_dups.csv`.

**Rótulo strict contra o rótulo original:**

| source_dataset | label_original | label | n |
|---|---|---|---|
| hatebr | hate | 1 | 693 |
| hatebr | neither | 0 | 3416 |
| hatebr | offensive_nothate | 0 | 2668 |
| pt_fortuna | 0 | 0 | 3879 |
| pt_fortuna | 1 | 1 | 1785 |
| toldbr | hate | 1 | 376 |
| toldbr | neither | 0 | 16771 |
| toldbr | offensive_nothate | 0 | 3650 |

Linhas fora do mapeamento esperado (hate e 1 → 1; neither, offensive_nothate e 0 → 0): 0. `label_policy`: {'strict': 33238}. `label_confidence`: {'medium': 20421, 'high': 12817}.

Taxa de ódio por fonte e split (estratificação):

| source_dataset | split | n | taxa_odio |
|---|---|---|---|
| hatebr | test | 968 | 0.1023 |
| hatebr | train | 4841 | 0.1023 |
| hatebr | val | 968 | 0.1023 |
| pt_fortuna | test | 809 | 0.3152 |
| pt_fortuna | train | 4046 | 0.3151 |
| pt_fortuna | val | 809 | 0.3152 |
| toldbr | test | 2971 | 0.0182 |
| toldbr | train | 14856 | 0.0181 |
| toldbr | val | 2970 | 0.0178 |

**Limpeza light** (proporção de textos):

| fonte | vazio | ate_2_tokens | so_user_url | perdeu_>50%_chars | tem_<user> | tem_<url> | tem_emoji_:nome: | text_clean_difere_da_limpeza_atual |
|---|---|---|---|---|---|---|---|---|
| hatebr | 0.0000 | 0.1139 | 0.0000 | 0.0006 | 0.0384 | 0.0007 | 0.2607 | 0.0000 |
| pt_fortuna | 0.0000 | 0.0000 | 0.0000 | 0.0037 | 0.4536 | 0.4200 | 0.0004 | 0.0000 |
| toldbr | 0.0000 | 0.0117 | 0.0000 | 0.0013 | 0.4473 | 0.1666 | 0.0993 | 0.0000 |
| total | 0.0000 | 0.0306 | 0.0000 | 0.0015 | 0.3650 | 0.1759 | 0.1153 | 0.0000 |

Comprimento mediano em caracteres: bruto 68, limpo 69. Taxa de ódio entre textos com até 2 tokens: 0.0492 (n = 1016).

## 2. Curva de aprendizado do linear PT

Receita de `scripts/pt_boost.py` (TF-IDF word 1-2 + char_wb 3-5, LogReg balanceada), treinada só em PT. Subamostra estratificada por fonte e rótulo, 3 sementes nas frações menores (dp = desvio-padrão entre sementes). Limiar ajustado na validação PT completa; test PT com 4748 textos.

| frac | n_treino | odio_treino | macro_f1 | macro_f1_dp | recall_odio | precisao_odio | pr_auc | pr_auc_dp | pr_auc_hatebr | pr_auc_pt_fortuna | pr_auc_toldbr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2500 | 5936 | 510 | 0.7070 | 0.0055 | 0.4616 | 0.4678 | 0.4313 | 0.0068 | 0.5077 | 0.5439 | 0.1683 |
| 0.5000 | 11873 | 1020 | 0.7279 | 0.0073 | 0.4845 | 0.5207 | 0.4761 | 0.0106 | 0.5815 | 0.5567 | 0.2198 |
| 0.7500 | 17807 | 1529 | 0.7304 | 0.0133 | 0.4575 | 0.5736 | 0.5043 | 0.0025 | 0.6259 | 0.5618 | 0.2585 |
| 1.0000 | 23743 | 2039 | 0.7572 | nan | 0.6225 | 0.5100 | 0.5233 | nan | 0.6546 | 0.5675 | 0.2897 |

## 3. Erros por fonte (test PT)

Linear da extensão = `tfidf_logreg_strict_s42` (bilíngue) com o Platt do bundle, nos níveis sensível e conservador. Linear PT = `pt_logreg_strict_s42`. Taxa de FP = FP sobre os não-ódio da fonte. PR-AUC e ROC-AUC não dependem do limiar. Colunas `_ic`: IC 95% por bootstrap (2000 reamostras estratificadas por fonte e classe, percentil).

| modelo | fonte | n | n_odio | recall | precisao | taxa_fp | fp | fn | pr_auc | roc_auc | recall_ic | precisao_ic | taxa_fp_ic | pr_auc_ic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear extensão (sensível 0,424) | todas | 4748 | 408 | 0.3211 | 0.6684 | 0.0150 | 65 | 277 | 0.5074 | 0.8886 | 0.275–0.365 | 0.606–0.727 | 0.012–0.019 | 0.460–0.555 |
| linear extensão (sensível 0,424) | hatebr | 968 | 99 | 0.3636 | 0.8182 | 0.0092 | 8 | 63 | 0.6092 | 0.8552 | 0.273–0.465 | 0.705–0.923 | 0.003–0.016 | 0.523–0.694 |
| linear extensão (sensível 0,424) | pt_fortuna | 809 | 255 | 0.3333 | 0.6800 | 0.0722 | 40 | 170 | 0.5817 | 0.7624 | 0.275–0.392 | 0.604–0.754 | 0.052–0.094 | 0.526–0.641 |
| linear extensão (sensível 0,424) | toldbr | 2971 | 54 | 0.1852 | 0.3704 | 0.0058 | 17 | 44 | 0.2349 | 0.8736 | 0.093–0.296 | 0.200–0.552 | 0.003–0.009 | 0.148–0.358 |
| linear extensão (equilibrado 0,57) | todas | 4748 | 408 | 0.2206 | 0.6977 | 0.0090 | 39 | 318 | 0.5074 | 0.8886 | 0.181–0.260 | 0.623–0.774 | 0.006–0.012 | 0.460–0.555 |
| linear extensão (equilibrado 0,57) | hatebr | 968 | 99 | 0.2525 | 0.8621 | 0.0046 | 4 | 74 | 0.6092 | 0.8552 | 0.172–0.343 | 0.731–0.969 | 0.001–0.009 | 0.523–0.694 |
| linear extensão (equilibrado 0,57) | pt_fortuna | 809 | 255 | 0.2353 | 0.6977 | 0.0469 | 26 | 195 | 0.5817 | 0.7624 | 0.184–0.290 | 0.602–0.792 | 0.031–0.065 | 0.526–0.641 |
| linear extensão (equilibrado 0,57) | toldbr | 2971 | 54 | 0.0926 | 0.3571 | 0.0031 | 9 | 49 | 0.2349 | 0.8736 | 0.019–0.185 | 0.118–0.625 | 0.001–0.005 | 0.148–0.358 |
| linear extensão (conservador 0,67) | todas | 4748 | 408 | 0.1593 | 0.7143 | 0.0060 | 26 | 343 | 0.5074 | 0.8886 | 0.125–0.196 | 0.620–0.806 | 0.004–0.008 | 0.460–0.555 |
| linear extensão (conservador 0,67) | hatebr | 968 | 99 | 0.2020 | 0.9524 | 0.0012 | 1 | 79 | 0.6092 | 0.8552 | 0.131–0.283 | 0.842–1.000 | 0.000–0.003 | 0.523–0.694 |
| linear extensão (conservador 0,67) | pt_fortuna | 809 | 255 | 0.1608 | 0.6949 | 0.0325 | 18 | 214 | 0.5817 | 0.7624 | 0.118–0.208 | 0.574–0.811 | 0.018–0.047 | 0.526–0.641 |
| linear extensão (conservador 0,67) | toldbr | 2971 | 54 | 0.0741 | 0.3636 | 0.0024 | 7 | 50 | 0.2349 | 0.8736 | 0.019–0.148 | 0.100–0.667 | 0.001–0.004 | 0.148–0.358 |
| linear PT (limiar val) | todas | 4748 | 408 | 0.6225 | 0.5100 | 0.0562 | 244 | 154 | 0.5233 | 0.9015 | 0.574–0.669 | 0.476–0.546 | 0.050–0.063 | 0.475–0.570 |
| linear PT (limiar val) | hatebr | 968 | 99 | 0.6465 | 0.5872 | 0.0518 | 45 | 35 | 0.6546 | 0.8842 | 0.545–0.737 | 0.513–0.667 | 0.038–0.067 | 0.570–0.738 |
| linear PT (limiar val) | pt_fortuna | 809 | 255 | 0.6392 | 0.5739 | 0.2184 | 121 | 92 | 0.5675 | 0.7634 | 0.580–0.694 | 0.530–0.621 | 0.184–0.253 | 0.515–0.628 |
| linear PT (limiar val) | toldbr | 2971 | 54 | 0.5000 | 0.2571 | 0.0267 | 78 | 27 | 0.2897 | 0.9165 | 0.370–0.630 | 0.193–0.327 | 0.021–0.033 | 0.196–0.427 |
| BERTimbau (limiar val) | todas | 4748 | 408 | 0.6250 | 0.5312 | 0.0518 | 225 | 153 | 0.5556 | 0.9136 | 0.576–0.672 | 0.496–0.568 | 0.046–0.059 | 0.509–0.606 |
| BERTimbau (limiar val) | hatebr | 968 | 99 | 0.6869 | 0.5965 | 0.0529 | 46 | 31 | 0.7115 | 0.9082 | 0.596–0.768 | 0.525–0.673 | 0.039–0.068 | 0.632–0.795 |
| BERTimbau (limiar val) | pt_fortuna | 809 | 255 | 0.6235 | 0.5719 | 0.2148 | 119 | 96 | 0.5757 | 0.7736 | 0.561–0.682 | 0.527–0.620 | 0.181–0.251 | 0.525–0.640 |
| BERTimbau (limiar val) | toldbr | 2971 | 54 | 0.5185 | 0.3182 | 0.0206 | 60 | 26 | 0.2782 | 0.8872 | 0.389–0.648 | 0.242–0.400 | 0.015–0.026 | 0.196–0.418 |

**BERTimbau menos linear PT, PR-AUC** (bootstrap pareado, mesmas reamostras):

| fonte | dif_pr_auc | ic95 | prop_reamostras_bertimbau_melhor |
|---|---|---|---|
| todas | 0.0324 | -0.013–0.082 | 0.9140 |
| hatebr | 0.0568 | -0.031–0.145 | 0.9030 |
| pt_fortuna | 0.0083 | -0.048–0.063 | 0.6110 |
| toldbr | -0.0116 | -0.115–0.101 | 0.4530 |

**Taxa de predito ódio por rótulo original.** Nas linhas `hate`/`1` é o recall; nas outras, a taxa de FP. Separa o FP sobre ofensa sem ódio do FP sobre texto neutro.

| fonte | rotulo_original | n | BERTimbau (limiar val) | linear PT (limiar val) | linear extensão (conservador 0,67) | linear extensão (equilibrado 0,57) | linear extensão (sensível 0,424) |
|---|---|---|---|---|---|---|---|
| hatebr | hate | 99 | 0.6869 | 0.6465 | 0.2020 | 0.2525 | 0.3636 |
| hatebr | neither | 494 | 0.0304 | 0.0385 | 0.0020 | 0.0040 | 0.0081 |
| hatebr | offensive_nothate | 375 | 0.0827 | 0.0693 | 0.0000 | 0.0053 | 0.0107 |
| pt_fortuna | 0 | 554 | 0.2148 | 0.2184 | 0.0325 | 0.0469 | 0.0722 |
| pt_fortuna | 1 | 255 | 0.6235 | 0.6392 | 0.1608 | 0.2353 | 0.3333 |
| toldbr | hate | 54 | 0.5185 | 0.5000 | 0.0741 | 0.0926 | 0.1852 |
| toldbr | neither | 2422 | 0.0169 | 0.0223 | 0.0021 | 0.0029 | 0.0054 |
| toldbr | offensive_nothate | 495 | 0.0384 | 0.0485 | 0.0040 | 0.0040 | 0.0081 |

## 4. LLM como juiz do rótulo

Amostra de 501 ids do test PT (`data/processed/auditoria_pt_amostra.csv`): erros e acertos comuns ao BERTimbau e ao linear PT. A amostra super-representa os erros, então o kappa com o rótulo vale para ela, não para o test inteiro. Prompt v1 de `scripts/llm_label.py`. Status por LLM: gemini (gemini-3-flash-preview): {'faltando': 441, 'ok': 60}; groq (openai/gpt-oss-120b): {'ok': 501}. Cada LLM é medido nos ids em que respondeu `ok`; as colunas LLM × LLM usam os 60 ids respondidos por todos.

**Concordância com o rótulo do corpus, por grupo.** Em FN, "discordar" quer dizer que o LLM também acha que não é ódio; em FP, que acha que é. A última coluna é a proporção em que os LLMs concordam entre si e discordam do rótulo.

| grupo | fonte | n gemini | concorda_rotulo gemini | n groq | concorda_rotulo groq | n ambos | LLMs_iguais_e_contra_rotulo |
|---|---|---|---|---|---|---|---|
| FN_comum | hatebr | 8 | 0.1250 | 20 | 0.0000 | 8 | 0.8750 |
| FN_comum | pt_fortuna | 9 | 0.3333 | 61 | 0.1803 | 9 | 0.4444 |
| FN_comum | todas | 19 | 0.2105 | 101 | 0.1881 | 19 | 0.6842 |
| FN_comum | toldbr | 2 | 0.0000 | 20 | 0.4000 | 2 | 1.0000 |
| FP_comum | hatebr | 2 | 1.0000 | 13 | 1.0000 | 2 | 0.0000 |
| FP_comum | pt_fortuna | 5 | 1.0000 | 62 | 0.8226 | 5 | 0.0000 |
| FP_comum | todas | 11 | 0.9091 | 100 | 0.7700 | 11 | 0.0000 |
| FP_comum | toldbr | 4 | 0.7500 | 25 | 0.5200 | 4 | 0.0000 |
| acerto_comum | hatebr | 4 | 0.2500 | 100 | 0.5000 | 4 | 0.7500 |
| acerto_comum | pt_fortuna | 9 | 0.8889 | 100 | 0.7400 | 9 | 0.1111 |
| acerto_comum | todas | 30 | 0.7333 | 300 | 0.7067 | 30 | 0.2333 |
| acerto_comum | toldbr | 17 | 0.7647 | 100 | 0.8800 | 17 | 0.1765 |

**Kappa de Cohen** (IC 95% por bootstrap, 2000 reamostras):

| fonte | n gemini | kappa rotulo × gemini | n groq | kappa rotulo × groq | n ambos | kappa LLM × LLM |
|---|---|---|---|---|---|---|
| todas | 60 | 0.248 [0.082–0.430] | 501 | 0.169 [0.096–0.242] | 60 | 0.624 [0.315–0.839] |
| hatebr | 14 | 0.041 [0.000–0.197] | 133 | -0.005 [-0.076–0.064] | 14 | 0.000 [0.000–0.000] |
| pt_fortuna | 23 | 0.427 [0.148–0.736] | 223 | 0.218 [0.112–0.318] | 23 | 0.679 [0.246–1.000] |
| toldbr | 23 | 0.291 [-0.082–0.646] | 145 | 0.379 [0.197–0.543] | 23 | 0.587 [-0.022–1.000] |

Leitura prevista no plano: se os LLMs concordam entre si e discordam do rótulo justamente nos FN e FP comuns, e não nos acertos, o teto é ruído de rótulo, não o algoritmo.
