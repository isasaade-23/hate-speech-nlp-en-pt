# Decisões metodológicas

Log append-only. Cada entrada registra o que foi decidido, quando, por quê e qual
alternativa foi rejeitada. Este arquivo vira a narrativa da seção de Métodos e a
munição para responder revisores. Não editar entradas antigas, apenas acrescentar.

Formato: `## [AAAA-MM-DD] Título` seguido de Decisão / Motivo / Alternativa rejeitada / Impacto.

---

## [2026-07-26] Escopo do projeto

**Decisão.** Classificador binário ódio/não-ódio, bilíngue inglês e português, apenas
texto. Detecção de idioma na entrada e roteamento para o modelo apropriado. Comparação
central entre modelos clássicos (dados tabulares de texto) e transformers.

**Motivo.** Os quatro datasets disponíveis cobrem inglês (Twitter e memes) e português
(Twitter). Não há dados nativos de Instagram ou Facebook, apesar do enquadramento inicial.
O produto será "texto de redes sociais", não específico por plataforma.

**Alternativa rejeitada.** Multimodal (imagem + texto dos memes). Rejeitada por custo e
complexidade; os memes entram apenas via texto OCR como dados extras em inglês.

**Impacto.** Modalidade text-only definida. Imagens dos datasets 1 e 4 não são extraídas.

---

## [2026-07-26] data/raw guarda apenas os arquivos tabulares extraídos

**Decisão.** Em vez de copiar os zips originais (cerca de 900 MB, majoritariamente
imagens) para data/raw, extrair somente os CSV de rótulos/texto. Os zips originais
permanecem em Downloads e são registrados com hash SHA-256 em data_provenance.md.

**Motivo.** Projeto text-only; as imagens não são usadas. Evita duplicar quase 1 GB.

**Alternativa rejeitada.** Copiar os zips inteiros (plano inicial). Rejeitada por
desperdício de disco sem ganho de reprodutibilidade (o hash + o membro extraído já
garantem rastreio até a fonte).

**Impacto.** data/raw fica pequeno e versionável em princípio; a proveniência é mantida
pelos hashes dos zips de origem.

---

## [2026-07-26] Codificação de cada dataset (achado crítico)

**Decisão.** Ler cada CSV com a codificação verificada empiricamente:
- dataset1 (Memotion): `utf-8-sig` (tem BOM).
- dataset2 (tweets EN): `utf-8` (apenas 3 U+FFFD em ~21 mil linhas, desprezível).
- dataset3 (PT, Fortuna): **`latin-1`**.
- dataset4 (MultiOFF): `utf-8-sig` (tem BOM).

**Motivo.** Sondagem de bytes (scratchpad/probe_encoding.py) mostrou que o dataset3
decodificado como utf-8 produz mojibake nos acentos ("n�o", "al�m"), enquanto em latin-1
sai português correto ("não", "além", "vê"). O arquivo é tecnicamente decodificável como
utf-8 sem erro, mas o resultado é semanticamente errado. cp1252 falha em 195 bytes
indefinidos; latin-1 nunca falha e preserva os acentos.

**Alternativa rejeitada.** Assumir utf-8 para todos. Rejeitada porque corromperia todos os
acentos do português, justamente o dataset mais limpo e importante do projeto.

**Impacto.** Qualidade do texto PT preservada. Decisão travada em configs/data.yaml.

---

## [2026-07-26] Contagem de linhas por parser CSV, não por quebras de linha

**Decisão.** Contar registros com um leitor CSV que respeita aspas, não por número de
linhas do arquivo.

**Motivo.** dataset2 tem cerca de 40,5 mil quebras de linha mas apenas ~21 mil registros,
porque muitos tweets contêm quebras de linha dentro de campos entre aspas. O mesmo vale
para dataset3.

**Impacto.** Loaders usam pandas/csv com parsing correto; contagens do inventário
confirmadas.

---

## [2026-07-26] Harmonização de rótulos: políticas strict e broad

**Decisão.** Materializar duas versões do corpus. `strict` mapeia para ódio apenas
categorias explicitamente de ódio; `broad` dobra ofensivo em ódio. Reportar ambas.

**Motivo.** Os datasets não compartilham a definição de "ódio". Transformar a
heterogeneidade em análise de sensibilidade sobre a fronteira ódio/ofensa é a principal
jogada de publicabilidade. Detalhes em label_mapping.md e configs/labels.yaml.

**Alternativa rejeitada.** Escolher um único mapeamento fixo. Rejeitada por esconder a
subjetividade do rótulo, que é o ponto que revisores atacam.

**Impacto.** MultiOFF sob `strict` só contribui negativos e por isso fica fora do corpus
primário sob strict (include_in_primary: false), entrando apenas sob broad.

---

## [2026-07-26] dataset2 (rótulos 1/2/3) entra sob gate de investigação

**Decisão.** O significado dos rótulos 1/2/3 do dataset2 não está documentado. Mapeamento
provisório inferido: 1 = ódio, 2 = ofensivo, 3 = neutro. A inclusão no corpus primário fica
condicionada (probe_gate) ao resultado do notebook 02_dataset2_label_probe (Fase 2).

**Motivo.** Distribuição (2.139 / 5.504 / 13.366, menor para maior) é compatível com a
ordem "ódio < ofensivo < nenhum" de corpora estilo Davidson, e a inspeção mostra a classe 1
saturada de slurs. Mas é inferência, não documentação oficial.

**Alternativa rejeitada.** Usar o mapeamento direto sem verificar. Rejeitada por risco de
rótulo errado contaminar o corpus primário.

**Impacto.** Escada de fallback: se a auditoria confirmar (kappa >= 0.6), usa o mapeamento;
se só a fronteira ódio-vs-resto for confiável, {1}->1 e {2,3}->0; se nada for confiável,
rebaixa dataset2 a conjunto auxiliar/teste externo em inglês.

---

## [2026-07-26] Ambiente Python

**Decisão.** venv com Python 3.12 (py -3.12) para a stack local de ML. Treino neural no
Colab controla sua própria stack (requirements-colab.txt).

**Motivo.** Máquina tem 3.13 (padrão) e 3.12; 3.11 não está instalado. 3.12 tem suporte mais
maduro em algumas libs de ML/DL do que o 3.13. requires-python do pacote é >=3.11.

**Impacto.** Reprodutibilidade local fixada em 3.12; pins exatos exportados por pip freeze
após validação do ambiente.

---

## [2026-07-26] Resultado do probe do dataset2 (Fase 2)

**Decisão.** Abrir o gate e incluir tweets_ip no corpus primário com o mapeamento
provisório 1=ódio, 2=ofensivo, 3=neutro (strict: {1->1, 2->0, 3->0}; broad: {1->1, 2->1,
3->0}). Marcado como `provisional_auto` até a auditoria manual.

**Motivo.** Evidência automática (reports/tables/dataset2_probe.csv):
- Ordem de distribuição 1<2<3 confirmada (2.139 / 5.504 / 13.366).
- Taxa de profanidade (léxico da better-profanity, pertencimento por token) cai
  monotonicamente: classe 1 = 0,553; classe 2 = 0,454; classe 3 = 0,209. A classe 1
  concentra a linguagem mais ofensiva, coerente com "ódio".
- Sinais de estilo (maiúsculas, exclamação) são planos entre classes, então a separação
  vem do conteúdo, não do estilo.

**Alternativa rejeitada.** Deixar tweets_ip fora até a auditoria manual. Rejeitada porque
a evidência automática é forte e o gate `provisional_auto` mantém a honestidade (revisável).

**Pendência.** Auditoria manual de 150 tweets em data/external/dataset2_audit_sample.csv
(preencher manual_label) para elevar a decisão a `confirmed` e calcular kappa. Se a
auditoria discordar, reverter via configs/labels.yaml (include_in_primary: false) ou
restringir a {1}->1, {2,3}->0.

**Impacto.** Corpus primário agora inclui os 21.009 tweets. Recompor harmonize/split/langid.

---

## [2026-07-27] Avaliação aprofundada (Fase 9): significância, calibração, transferência

**Decisão 1: Significância pareada por McNemar + correção de Holm.** Comparar modelos
por McNemar exato (binomial) sobre as predições por exemplo, não por diferença de
macro-F1 pontual. As comparações são feitas SÓ dentro de uma política (strict/broad têm
linhas e rótulos de teste diferentes; um teste pareado entre elas não tem sentido). Dentro
da política, todos os modelos usam o mesmo split congelado, então as predições alinham por
`id`. Múltiplas comparações por política são corrigidas por Holm-Bonferroni (α=0,05).

**Motivo.** Diferenças de 1-2 pontos de macro-F1 podem não ser significativas; o revisor
vai exigir o teste. McNemar é o padrão para classificadores pareados no mesmo conjunto.

**Decisão 2: Predições por exemplo persistidas (reports/predictions/).** train.py grava
(id, y_true, y_score, y_pred) por split; modelos já treinados são reconstruídos do joblib
congelado + corpus congelado, sem re-treinar. É o substrato de McNemar e da calibração.

**Decisão 3: Calibração reportada (ECE/MCE/Brier + diagrama de confiabilidade).** Scores
passam por min-max antes dos bins para que o decision_function do SVM fique no mesmo eixo
[0,1] das probabilidades (monotônico, não altera ranking/AUC). Uma figura por política.

**Motivo.** O produto expõe um score; calibração diz se ele é interpretável como
probabilidade. É requisito do "Reproducibility/Ethics statement" e da seção de produto.

**Decisão 4: Trilha SBERT (embeddings densos multilíngues).** Adicionar
`paraphrase-multilingual-MiniLM-L12-v2` (encoder CONGELADO, sem fine-tuning) →
LogReg/LightGBM, como baseline clássico forte que compartilha a ideia multilíngue do
XLM-R sem GPU. O encoder congelado torna o anti-leakage automático (nada é aprendido de
nenhum split); embeddings são cacheados por hash-de-texto (LogReg e LGBM reusam a matriz).

**Motivo.** (a) LightGBM rendia mal e lento sobre TF-IDF esparso de 60k dim; 384 dim
densos são seu habitat natural. (b) TF-IDF de palavra não compartilha vocabulário entre
idiomas. É necessário para a transferência cross-lingual.

**Decisão 5: Experimento de transferência cross-domínio e cross-lingual (destaque).**
Treinar numa fatia e testar em fatia disjunta: cross-domínio EN (tweets↔memes) e
cross-lingual zero-shot (EN→PT e PT→EN). Protocolo idêntico ao principal: features no
treino apenas, limiar ajustado num val da fonte de treino, teste único na fonte-alvo. Como
treino e teste são datasets diferentes, não há leakage na fronteira. Roda com TF-IDF E
SBERT. O contraste (TF-IDF colapsa cross-lingual, SBERT transfere) é o resultado.

**Alternativa rejeitada.** Reportar só números in-distribution. Rejeitada: mede o risco de
domain shift de forma indireta; a transferência explícita é medida direta e vira a Fig. 5.

---

## [2026-07-27] CORREÇÃO: encoding do pt_fortuna era UTF-8, não latin-1 (bug)

**Decisão.** Trocar o encoding do dataset3 (pt_fortuna) de `latin-1` para `utf-8` em
configs/data.yaml e reconstruir todo o pipeline a jusante (ingest → harmonize → split →
langid → train → report → analyze → transfer).

**Motivo.** Prova por bytes crus, não por glifo de console. Para a palavra "parabéns" o
arquivo contém `70 61 72 61 62 C3 A9 6E 73`. `C3 A9` é a codificação UTF-8 de `é`
(U+00E9). Decodificado como UTF-8 dá `parabéns` (é = ord 233); decodificado como latin-1
dá `parabÃ©ns` (Ã=195, ©=169). Contagem de marcadores de mojibake (Ã/Â) no arquivo:
latin-1 → 12.045; utf-8 → ~257 (esses são ã/â legítimos do português). O arquivo é UTF-8
válido em toda a extensão (decodifica sem erro). Portanto `latin-1` corrompia TODO acento
do português: é→Ã©, ã→Ã£, ç→Ã§, etc., e a limpeza de controles (`\x7f-\x9f`) ainda comia
o segundo byte, deixando "PARABÉNS" como "PARAB� NS".

**Correção de uma decisão anterior.** A entrada de [2026-07-26] sobre latin-1 estava
ERRADA (o teste empírico original provavelmente se enganou com a renderização do console
do Windows, que não exibe acentos e mostra `latin-1` "parecendo" mais legível). Regra:
validar encoding por codepoint/byte, nunca por glifo de terminal.

**Impacto.** (1) O TF-IDF de char aprendeu a corrupção de forma consistente, então os
números PT in-distribution mudam pouco (F1 ~0,64-0,69). (2) O SBERT e a transferência
cross-lingual eram os mais prejudicados (o tokenizer multilíngue via `parabÃ©ns` como
lixo). Espera-se melhora após a correção. (3) Cache de embeddings por-texto: os textos
PT corretos são chaves novas (re-encodadas ~5,6k), o EN todo continua cache hit. (4)
Splits recongelados (o conteúdo PT mudou; split_sha256 novo). Resultados da Fase 9
recomputados sobre texto correto. Aprendizado para limitations.md: auditar encoding por
bytes é parte do protocolo de proveniência.

---

## [2026-07-27] Modelo de produto por Pareto (Fase 11)

**Decisão.** Selecionar o modelo servido por uma regra de Pareto multi-eixo, não pelo topo
de macro-F1. Recomendação por perfil: XLM-R para qualidade (GPU, research-only);
**tfidf_logreg como MVP CPU** (p95 1,6 ms, 3,6 MB, autossuficiente, F1 a ~4 pts do XLM-R);
sbert_lgbm se o score calibrado for requisito (melhor ECE, mas +470 MB de encoder). Tabela
em `hsc product` → reports/tables/product_selection_*.csv; racional em product_decision.md.

**Motivo.** Um produto pondera qualidade contra latência, tamanho, calibração, viés e,
decisivamente, licença. O melhor F1 (XLM-R) é neural (GPU) e research-only; o clássico
minúsculo perde ~4 pts mas serve em CPU sem dependências pesadas.

**Alternativa rejeitada.** Servir o de maior macro-F1 direto. Rejeitada por custo de
deploy (GPU, ~1,1 GB) e pela trava de licença (corpus completo = research-only).

**Impacto (2 fixes de deploy).** (1) `EmbeddingVectorizer.__getstate__` deixou de picklar o
encoder de ~470 MB → joblib do sbert_logreg_strict caiu de 479 MB para 3 KB (recarrega sob
demanda). (2) `inference.py` passou a escolher o melhor joblib LOCAL servível. Antes
tentava o XLM-R (melhor F1 global, pesos no Colab) e quebrava a API; default agora é
tfidf_logreg_strict.

---

## [2026-07-27] Release público no GitHub

**Decisão.** Publicar o repositório como público em
github.com/isasaade-23/hate-speech-nlp-en-pt, com README de portfólio em inglês (sem
emojis), capa SVG na identidade visual, figuras-herói versionadas em assets/, CITATION.cff
e topics. Código sob AGPL-3.0 desde 24/08/2026 (antes MIT): copyleft de rede, para que
quem rodar uma versão modificada como serviço devolva o código. Pesos do modelo ficam fora
da AGPL, em LICENSE-MODEL.md, porque as licenças de dados não permitem conceder uso
comercial (data_provenance.md).

**Motivo.** Artefato de portfólio para candidaturas; recrutadores precisam ver o repo.
Nenhum dado ou segredo é versionado (data/, .env, pesos são gitignored) → público é seguro.

**Impacto.** API validada localmente por uvicorn antes do release (EN + PT com acento).
Fase 12 (Docker) fica pendente do WSL2 (Windows 11 Home, erro Wsl/0x80070422): habilitar
os recursos WSL + VirtualMachinePlatform e reiniciar; os arquivos de deploy já estão prontos.

---

## Decisão: NÃO remover stop words (28/07)

**Decisão.** Manter o pré-processamento sem remoção de stop words. Suporte a `stop_words`
(lista bilíngue EN+PT) fica no código como opção (`src/hsc/features/stopwords.py`,
configs `*_nostop.yaml`), desligado por padrão.

**Motivo.** Ablação em toda a família TF-IDF (logreg/lgbm/svm × strict/broad): ΔF1 de −0.009 a
+0.005 (média ≈ −0.4 pt) e **AUC praticamente idêntica** (ΔAUC ≈ −0.06 pt) → **não ajuda, e às
vezes atrapalha**. Char n-grams + IDF já tratam palavras funcionais; remover pronomes/preposições
descarta dêixis útil pra ódio. A lista omite "no" (preposição PT = negação EN). Tabela completa em
experiment_log.md.

**Impacto.** Ablação negativa documentada (justifica a escolha de pré-processamento para o
revisor). Nenhuma mudança no modelo servido.

---

## Decisão: incorporar HateBR como 5ª fonte (PT, Instagram) (11/08)

**Decisão.** Adicionar o HateBR (Vargas et al. 2022) ao corpus de pesquisa como `hatebr`:
7.000 comentários de Instagram em PT-BR, anotados por 3 especialistas. Loader
`src/hsc/ingest/dataset5_hatebr.py`; mapeamento em `configs/labels.yaml`. Fonte research-only
(Sinch) → **fora** do `commercial_whitelist` (produto não usa).

**Motivo.** Ataca três fraquezas de uma vez: (1) PT deixa de ser fonte única — antes só
`pt_fortuna` (5.670, comentário web), agora ganha um segundo domínio (Instagram); (2)
desconfunde idioma×domínio no PT; (3) o HateBR traz a fronteira ofensa≠ódio **anotada na
origem**, não inferida como no `tweets_ip`.

**Mapeamento (o ponto científico).** O rótulo binário do HateBR é `offensive_language`, que é
ofensa, não ódio. A coluna `hate_speech` é um código de categoria: `0` = não-ofensivo,
`-1` = ofensivo mas não-ódio, e códigos 1–9 (mais frações de múltiplas categorias) = ódio real.
O loader dobra isso numa `label_original` de 3 vias {neither, offensive_nothate, hate}, e a mesma
coluna alimenta as duas políticas: **strict** marca 1 só em `hate` (702 positivos, ~10%);
**broad** dobra ofensivo em ódio (= `offensive_language`, 3.500 positivos). Assim o HateBR reforça
exatamente o eixo strict/broad em vez de poluí-lo. Confiança `high` nas 3 categorias (anotação
especialista com concordância Kappa/Fleiss reportada pelos autores).

**Domínio.** Instagram entra como `web_comment` (mesmo registro de comentário curto de usuário
do `pt_fortuna`); as fontes seguem separáveis por `source_dataset` nas quebras. Não foi criado um
domínio novo no schema para não fragmentar a estratificação.

**Codificação.** `HateBR.csv` (franciellevargas/HateBR @2d18c5b9) é UTF-8 sem BOM, 0 caracteres de
substituição — verificado antes de ingerir (lição do susto latin-1 do `pt_fortuna`).

**Impacto.** Reabre dedup + re-congelamento do split → **todo o leaderboard v1 fica obsoleto**
(clássicos + neurais foram treinados no corpus sem HateBR). O corpus v1 foi preservado em
`data/processed/_pre_hatebr_v1/`. Modelos precisam ser retreinados no corpus v2 para
comparabilidade. ToxSyn-PT (sintético) fica para experimento de robustez separado, fora do
corpus primário e fora do test.

## Decisão: calibrar o score servido com Platt scaling (14/08)

**Contexto.** O demo exibe o score do `tfidf_logreg_strict_s42` rotulado como "hate
probability", mas o score cru não era uma probabilidade honesta: ECE 0.156 no teste (um texto
mostrado com "90%" acertava bem menos que 90% das vezes).

**Experimento (`scripts/calibrate_demo.py`).** Platt (sigmoide sobre o score) e regressão
isotônica, ambos com fit APENAS na validação; teste tocado uma vez por método.
Resultado (`reports/tables/calibration_demo.csv`): ECE 0.156 → 0.030 (Platt) / 0.021
(isotônica); Brier 0.111 → 0.077; macro-F1 do Platt idêntico (0.7293) porque a sigmoide é
estritamente monótona e o limiar é mapeado exatamente (0.6357 → 0.3168). A isotônica marcou
0.7332 por colapsar empates na fronteira — ganho frágil, dentro do ruído.

**Decisão.** Platt no bundle servido: `bundle["calibration"] = {coef, intercept, threshold}`,
aplicado por `HateClassifier` quando presente (`scripts/fit_platt_demo.py`). Escolhido sobre a
isotônica por ser suave (a barra de confiança do demo não salta em degraus), preservar o F1
exatamente e caber em dois números. O score cru, o limiar cru e o registry ficam intocados, então
toda análise existente (McNemar, predições salvas) permanece reproduzível.

**Alternativa rejeitada.** `CalibratedClassifierCV` do sklearn: refitaria o estimador em folds
internos, mudando o modelo comparado no leaderboard. Aqui o modelo é o mesmo; só a escala do
score muda.

## Decisão: incorporar ToLD-Br como 6ª fonte (PT, Twitter) → corpus v3 (14/08)

**Por quê.** Depois de três manobras medidas com efeito zero (HPO bayesiano, limiar por idioma,
ablação de normalização) e uma negativa (embeddings congelados do LFM2.5-Encoder: teste 0.6698 vs
0.7293 do TF-IDF), mais dados PT é a alavanca restante de maior expectativa. ToLD-Br (Leite et
al. 2020): 21.000 tweets PT, 3 anotadores leigos × 6 categorias (contagens 0..3).

**Mapeamento (o ponto científico).** Voto majoritário (≥2 de 3) dobrado no MESMO 3-way do
HateBR: categorias dirigidas a identidade (homofobia, racismo, misoginia, xenofobia) → `hate`
(376, 1.8%); obsceno/insulto → `offensive_nothate` (3.687); resto → `neither` (16.937). strict
marca 1 só em `hate`; broad dobra ofensivo. Voto minoritário (1 de 3) desce de balde — anotação
leiga com concordância moderada, daí confiança `medium` fora de `hate`.

**Desconfusão idioma×domínio.** Primeiro tweet em PT do corpus: até aqui PT era só web_comment
(pt_fortuna, hatebr) e tweet era só EN. As quebras por fonte agora separam os dois eixos no PT.

**Licença.** Dados CC BY-SA 4.0 (código MIT). Fora da `commercial_whitelist`: share-alike sobre
pesos derivados é terreno jurídico não testado. 109 U+FFFD nativos em 21k linhas (artefato da
fonte, mantido; auditado como no susto latin-1).

**Impacto.** Corpus v3: strict 61.671 linhas brutas → 60.473 após dedup (hate=5.226 pré-dedup),
splits 43.195/8.639/8.639, hash 229603f7dd5f; broad 61.214, hash 6a41ef621c40. **Leaderboard v2
obsoleto** (clássicos retreinados localmente; neurais precisam de re-run no Colab). v2 preservado
em `data/processed/_pre_toldbr_v2/`. Números v3 NÃO são comparáveis aos v2: o teste mudou.

**Resultado do retreino clássico v3 (14/08, 16 runs, log `_v3_classical_retrain.log`).**
`tfidf_logreg` segue o melhor clássico nas duas políticas: strict 0.7061 (recall-ódio 0.529),
broad 0.7311 (recall-ódio 0.618). Ordem da família preservada (logreg > lgbm > svm; TF-IDF >
SBERT congelado). Platt refitado na validação v3: ECE de teste 0.155 → 0.026, limiar
0.5765 → 0.2173 calibrado, F1 intacto. No demo, o exemplo PT de misoginia subiu de p=0.38
(raspando o limiar v2) para p=0.68 — o efeito direto dos 21k tweets PT. Transformers ainda
são v2: precisam de re-run no Colab antes de qualquer comparação clássico×neural em v3.

## Beta 2.0: memotion vira teste externo + ensemble empilhado servido (14/08)

**Motivação.** Meta da usuária: AUC ≥ 0.9 no servido, sem vazamento. Guia: o survey
Gandhi et al. 2024 (Expert Systems, DOI 10.1111/exsy.13562), que aponta ensembles e
features de léxico afetivo como as alavancas com ganho documentado, e não inventaria
nenhum corpus PT (nossos 3 corpora PT estão fora do radar da revisão).

**Diagnóstico que redefiniu o escopo.** Quebra por fonte da AUC do servido no teste v3:
hatebr 0.90, toldbr 0.88, tweets_ip 0.84, pt_fortuna 0.74 e **memotion 0.547 — aleatório**.
O texto OCR de meme sozinho não carrega o ódio (o survey mede o mesmo colapso, 0.48-0.53,
em modelos text-only sobre memes). Decisão da usuária: memotion sai do corpus primário
(`include_in_primary: false`, 2 políticas) e fica guardado como teste externo.

**Corpus v4.** strict: 54.679 brutas → 53.540 após dedup, splits 38.241/7.650/7.649,
hash 1657bf97bb5f. broad: 54.281, hash 1144ef915567. Portão de leakage verde.

**Ensemble (scripts/build_stack.py).** Meta-LogReg sobre os scores dos clássicos; meta,
limiar e Platt fitados APENAS na validação; teste tocado 1× por composição.

| Composição (strict, teste v4) | ROC-AUC | macro-F1 | recall-ódio | ECE |
|---|---|---|---|---|
| tfidf_logreg solo | 0.8728 | 0.7022 | 0.481 | — |
| stack 3×TF-IDF (SERVIDO) | 0.8772 | 0.7105 | 0.548 | 0.044 |
| stack 5 (+2 SBERT) | 0.8864 | 0.7153 | 0.511 | 0.043 |

Servimos o 3×TF-IDF (15 MB, 34 ms/texto em CPU); o de 5 ganha +0.009 AUC mas exige o
encoder SBERT de 470 MB, inviável no 1 GB do Streamlit free. Registrado como trade-off,
não descartado. Por fonte (stack servido): hatebr 0.912, toldbr 0.906, tweets_ip 0.829,
pt_fortuna 0.746 — o gargalo agora é o EN de tweets_ip e a subjetividade do pt_fortuna.

**Suporte de inferência.** `HateClassifier` aceita bundle `kind='stack'` (membros inline,
coeficientes do meta, Platt); o membro linear vira a superfície de explicação do demo,
então a atribuição por termos segue exata para essa parcela do score.

**Rumo ao 0.9.** Faltam ~0.02 de AUC. Próximas alavancas (fase seguinte da Beta 2.0):
HurtLex/features afetivas nos modelos-base (ganho documentado no survey), datasets
Vidgen "Dynamically Generated" (41k, >54% ódio) e HateXplain, e o destilado ONNX int8
treinado no Colab como quarto membro. Números v4 NÃO são comparáveis a v1/v2/v3 (teste
mudou); transformers seguem reportados como estudo anterior até re-run no Colab.

---

## 2026-08-16 — Corpus v5 (Beta 2.0, fase 2): Vidgen + HateXplain + HurtLex

**Novas fontes.** Vidgen et al. 2021 "Dynamically Generated" (41.144 entradas EN
sintéticas adversariais, 54% ódio, CC BY 4.0) e HateXplain (20.148 posts EN
Twitter+Gab, 3 anotadores com rótulo 3-way nativo, MIT; 919 empates 3-way
descartados). As duas entram na whitelist comercial — primeiras fontes de texto
com licença permissiva além do multioff.

**Perda dos zips-fonte 1–4.** Verificado em 16/08: os zips originais dos datasets
1–4 não existem mais em nenhum disco local. Os arquivos extraídos em data/raw
sobrevivem com proveniência registrada; o ingest agora os reutiliza com a flag
explícita `zip_absent_reused_raw` carregando o sha256 gravado antes da perda.

**Anti-leakage reforçado no split.** O near-dup por char-3gram (0,90) capturou só
~1,1k dos ~15k pares original/perturbação do Vidgen. Os links `acl.id.matched` do
próprio dataset agora são unidos aos clusters antes do split por grupo: 14.443
links fundidos, ZERO pares cruzando splits (antes eram 6.554 — dos quais 6.545
com rótulos opostos, ou seja, risco de inflação ~nulo, mas a regra é regra).

**Corpus v5 strict:** 115.052 linhas → 113.826 pós-dedup; splits 81.304/16.261/
16.261, hash `9bfda377058d`. Broad: 114.567, hash `0921bb2ce617`. Ódio no strict:
28,8% (era 8% no v4) — desbalanço resolvido pelo Vidgen.

**Capacidade.** Com o corpus 2×, o teto de 50k features/bloco estava limitando:
sweep no val deu 150k = vocabulário inteiro (256.921 dims), val AUC 0,8827→0,8843.
Teto elevado para 150k nos 3 membros (min_df=2 não ajudou).

**Resultados (strict, teste v5 — NÃO comparável ao v4: o teste agora contém 5.871
exemplos adversariais do Vidgen, 36% do total):**

| Modelo (teste v5) | ROC-AUC | macro-F1 | recall-ódio | ECE |
|---|---|---|---|---|
| tfidf_logreg solo (150k) | 0.884 | 0.788 | 0.707 | — |
| stack 3×TF-IDF (150k) | **0.8875** | **0.7906** | **0.7359** | 0.0625 |
| — só texto real (sem vidgen) | 0.8705 | — | — | — |

Por fonte (stack): hatexplain 0.880, toldbr 0.869, hatebr 0.863, vidgen 0.810,
pt_fortuna 0.762, tweets_ip 0.730. Recall de ódio saltou 0,55→0,74 — o efeito
produto mais visível do v5.

**Regressão PT e a variante lang_bal.** As fontes PT caíram vs v4 (hatebr
0,912→0,863; toldbr 0,906→0,869): os ~60k textos EN novos diluíram o PT.
Experimento no val: pesos amostrais balanceando idioma×classe recuperam o PT
(hatebr 0,872→0,895; toldbr 0,855→0,870 no val) ao custo de −0,006 no AUC global
e −0,004 no EN. DECISÃO EM ABERTO (produto): servir o baseline (métrica global,
meta 0,9) ou a variante PT-first (identidade da Luciola). Registrado, não decidido.

**Negativos novos (não repetir).** (1) Bloco HurtLex (36 dims, léxico EN+PT
CC BY-NC-SA — NonCommercial, nunca no modelo de produto): val AUC 0,8827→0,8829,
nada; como 4º membro do stack: AUC idêntico. Transformer fica no repo como
registro. (2) min_df=2: nada.

**Rumo ao 0.9.** No teste v5 (adversarial-heavy) faltam 0,0125. Alavancas que
restam: membros SBERT/destilado re-treinados no Colab (no v4 os 2 SBERT davam
+0,009), e o re-run dos transformers. O número 0,9 precisa ser lido contra o
teste que o contém: o v5 é substancialmente mais difícil que o v4.

---

## 2026-10-06 — Etapa 1: repositório pronto para outra pessoa rodar

**Objetivo.** Deixar o repositório executável do zero por outra pessoa (cenário:
o Vinícius clonando agora), não só pela autora.

**Bug de portabilidade achado e corrigido.** `configs/data.yaml` fixava
`source_dir` em `C:/Users/Renato/Downloads/hate-speech`, um caminho absoluto da
máquina da autora, e `hsc/ingest/__init__.py` lia esse valor direto com
`Path()`, sem passar por `resolve()`. Qualquer outra pessoa rodando `hsc ingest`
cairia nesse caminho inexistente. Corrigido: `source_dir` agora é
`data/source_zips` (relativo, resolvido contra a raiz do projeto como todo o
resto), e o ingest passou a usar `resolve()` como os demais caminhos. Pasta
`data/source_zips/` criada (vazia, só `.gitkeep`) e adicionada ao `.gitignore`
junto com as demais pastas de dado.

**Verificação.** Rodado `hsc ingest` e `hsc harmonize --policy strict` com o fix
aplicado, sem os zips-fonte presentes (usando o fallback de reaproveitar os
arquivos já extraídos em `data/raw`, documentado desde 16/08): as 8 fontes
processaram normalmente, corpus strict saiu com 115.052 linhas (hate=33.115),
igual ao de antes do fix. Sem regressão.

**`data/MANIFEST.csv` criado**, com as 8 fontes (fonte, idioma, nº de textos,
licença, url de origem, `commercial_ok`, observação). URLs de pt_fortuna,
multioff, hatebr, toldbr, vidgen e hatexplain conferidas por HTTP (200) nesta
sessão; a de multioff estava registrada errada de memória
(`.../Multimodal-Meme-Classification-Hate-Speech-Detection`, 404) e foi
corrigida para o repositório real
(`.../Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text`).
memotion e tweets_ip seguem com licença/URL pendente de confirmação, como já
constava em `data_provenance.md`.

**2.1 implementado: flag de corpus comercial.** `build_corpus()` (harmonize.py)
ganhou `commercial_only` e `include_toldbr`; CLI ganhou `hsc harmonize
--commercial` e `--include-toldbr`. A flag só RESTRINGE fontes já marcadas
`include_in_primary` pela policy (strict/broad) — nunca promove uma fonte que a
policy já exclui. Testado nas duas policies: sob `strict`, `--commercial` dá
60.373 linhas (vidgen+hatexplain; multioff fica fora porque sob strict ele já
não tem rótulo de ódio confirmado, regra pré-existente do labels.yaml); sob
`broad`, multioff entra e dá 61.116 linhas; com `--include-toldbr` adicional,
81.373 linhas. Arquivos de saída ganham sufixo (`corpus_strict_commercial.parquet`,
`corpus_strict_commercial_toldbr.parquet`) para não sobrescrever o corpus
completo.

**README.** Adicionada a seção "Current status" no topo (corpus v5, stack v5
ainda servido, twitter-XLM-R escolhido pelo gate de viés de 24/08 mas bloqueado
por tamanho — ver `pt_recall_x_vies.md`) e "Próximos passos (Etapa 2)" com os
itens 2.1–2.7, marcando 2.1 e 2.2 como feitos. Quickstart reescrito: passo
explícito de download das 8 fontes (link de cada uma em `data/MANIFEST.csv`,
nome do zip esperado) antes do `hsc ingest`, e os dois comandos novos de
`--commercial`.

**Pendente (Etapa 2.2):** retreinar o baseline no corpus comercial e comparar
com o atual no teste congelado (ROC-AUC, macro-F1, recall por idioma/fonte,
sonda de identidade de 192 frases, strict e broad). Não executado ainda nesta
sessão — fica para o próximo passo, com o resultado (positivo ou negativo)
registrado aqui depois.

---

## 2026-10-07 — INCIDENTE: split congelado do strict apagado e restaurado

**O que aconteceu.** Ao testar a flag `--commercial` na sessão da Etapa 1
(06/10), rodei `hsc harmonize --policy strict` (sem `--commercial`) para
conferir a trilha normal antes de testar a nova. Não tinha registrado que
`build_corpus()` **sobrescreve** `data/processed/corpus_strict.parquet` do
zero, sem as colunas que `hsc split` e `hsc langid` tinham adicionado depois
da última vez que o corpus foi harmonizado. `corpus_broad.parquet` não foi
tocado (só rodei harmonize nele via `--commercial`, que grava em arquivo
separado) e permaneceu intacto o tempo todo.

**Descoberto ao rodar a Etapa 2.2**: `build_variant()` falhou com
`KeyError: 'split'`. Investigação confirmou: `corpus_strict.parquet` tinha
voltado a 115.052 linhas sem dedup e sem as colunas `dup_cluster_id`/`split`/
`lang_pred`. Nenhum backup em git (`data/processed/` é propositalmente
gitignored).

**Reconstrução, verificada antes de escrever qualquer coisa.**
1. `drop_exact_duplicates` (determinístico, sem aleatoriedade) no corpus
   corrompido: 115.052 → 113.826 linhas, removendo exatamente 1.226 —
   os mesmos números do manifesto congelado.
2. ids de teste e validação recuperados de `reports/predictions/*_strict_s42_*.parquet`
   já salvos (8 modelos diferentes, todos com os MESMOS 16.261 ids de teste e
   16.261 de validação, sem sobreposição entre si) — `tfidf_logreg`, `stack`,
   `twitter_xlmr`, `xlmr_multilingual`, `tfidf_lgbm`, `tfidf_svm`,
   `tfidf_logreg_hurtlex`, `lfm25_encoder`. Todo o resto virou `train`
   (81.304 linhas).
3. **Prova, não suposição**: `data/processed/splits_strict.json` (nunca
   tocado) guarda o hash SHA-256 do split original. Recalculei o hash da
   minha reconstrução pela mesma fórmula (`id:split` por linha, concatenado,
   SHA-256) e bateu **exatamente**:
   `9bfda377058d9264339179e02d066b771f9ed722602a3b529087bafa4a3651ae`.
   Só depois desse match escrevi o arquivo de volta.

**O que ficou diferente do original:** `dup_cluster_id` não foi restaurado
(não precisava para filtrar o treino por fonte, e recalculá-lo tem risco de
pequenas diferenças de ordenação que eu preferi não correr sem necessidade).
Se algum dia for preciso de novo — outro re-split, por exemplo — ele pode ser
recomputado determinística e independentemente a partir dos mesmos dados
brutos, que não mudaram.

**Causa raiz.** `hsc harmonize` e `hsc split` escrevem no MESMO arquivo
(`corpus_<policy>.parquet`), e o primeiro não verifica nem avisa se o segundo
já rodou em cima dele. Rodar harmonize de novo por engano apaga o split sem
erro nem aviso.

**Correção de processo, não só de dado:** a partir de agora, antes de rodar
`hsc harmonize --policy X` (com ou sem `--commercial`) em qualquer sessão,
confiro primeiro se `corpus_X.parquet` já tem a coluna `split` — se tiver,
é sinal de que already passou por `hsc split` e sobrescrever é destrutivo.

---

## 2026-10-07 — Etapa 2.2: baseline no corpus comercial, resultado negativo esperado

**Protocolo.** Nenhum split foi refeito. `scripts/commercial_retrain_compare.py`
filtra o corpus JÁ SPLITADO: mantém validação e teste intactos (todas as
fontes, exatamente como no modelo servido) e restringe só o TREINO às fontes
`commercial_ok` (`configs/labels.yaml: commercial_whitelist` = multioff,
vidgen, hatexplain). Duas variantes por policy: com e sem ToLD-BR (CC BY-SA
4.0, licença ainda pendente, opt-in separado). Baseline treinado:
`tfidf_logreg` (o config do Quickstart), comparado contra o próprio
`tfidf_logreg` no corpus completo e contra o stack v5 servido, os três
avaliados line a line no MESMO teste congelado.

**Textos em PT que sobram no treino, antes de treinar (como pedido):**

| variante | treino total | treino PT | ódio PT no treino |
|---|---|---|---|
| strict, commercial | 43.059 | **0** | 0 |
| strict, commercial + ToLD-BR | 57.915 | 14.856 | 269 |
| broad, commercial | 43.589 | **0** | 0 |
| broad, commercial + ToLD-BR | 58.444 | 14.855 | 2.875 |

Sob `strict`, as duas únicas fontes comerciais (Vidgen, HateXplain) são 100%
em inglês — o treino comercial sem ToLD-BR fica **sem nenhum texto em
português**. Isso já era esperado antes de rodar qualquer treino.

**Resultado no teste congelado (strict):**

| modelo | ROC-AUC | macro-F1 | recall ódio | recall EN | recall PT | memotion AUC (externo) |
|---|---|---|---|---|---|---|
| stack v5 (servido) | 0,8875 | 0,7906 | 0,7359 | 0,7757 | 0,3162 | — |
| tfidf_logreg (corpus completo) | 0,8854 | 0,7878 | 0,7073 | 0,7439 | 0,3211 | 0,5233 |
| tfidf_logreg (comercial) | 0,8416 | 0,7686 | 0,6363 | 0,6954 | **0,0123** | 0,4912 |
| tfidf_logreg (comercial + ToLD-BR) | 0,8479 | 0,7699 | 0,6517 | 0,7119 | **0,0172** | 0,4980 |

**Resultado no teste congelado (broad):**

| modelo | ROC-AUC | macro-F1 | recall ódio | recall EN | recall PT | memotion AUC (externo) |
|---|---|---|---|---|---|---|
| stack v5 (servido) | 0,8396 | 0,7561 | 0,7192 | 0,7681 | 0,4989 | — |
| tfidf_logreg (corpus completo) | 0,8352 | 0,7521 | 0,7247 | 0,7710 | 0,5164 | 0,5044 |
| tfidf_logreg (comercial) | 0,7564 | 0,7040 | 0,5937 | 0,7129 | 0,0572 | 0,4963 |
| tfidf_logreg (comercial + ToLD-BR) | 0,7982 | 0,7217 | 0,6346 | 0,7231 | 0,2365 | 0,4996 |

**A queda em português é grande, como esperado, e ToLD-BR não resolve.** Mesmo
com 14.856/14.855 textos PT no treino graças ao ToLD-BR, o recall PT comercial
fica muito abaixo do corpus completo nas duas policies (strict: 0,0172 contra
0,3211; broad: 0,2365 contra 0,5164). Sob strict, ToLD-BR tem só 269 positivos
de ódio em 14.856 linhas (1,8%) — pouco sinal para compensar a ausência de
HateBR e pt_fortuna. Sob broad os positivos de ToLD-BR saltam para 2.875
(offensive vira ódio), o que explica o salto maior de recall PT ali (0,0572 →
0,2365) comparado ao strict (0,0123 → 0,0172).

**Achado que pede cuidado na leitura: a sonda de identidade "melhora" nos
modelos comerciais, mas não por serem menos enviesados.**

| modelo (strict) | FP de fundo PT | FP com identidade | excesso |
|---|---|---|---|
| stack v5 (servido) | 0,0141 | 0,1510 | 0,1370 |
| tfidf_logreg (corpus completo) | 0,0150 | 0,1354 | 0,1204 |
| tfidf_logreg (comercial) | 0,0088 | 0,0365 | 0,0277 |
| tfidf_logreg (comercial + ToLD-BR) | 0,0021 | 0,0000 | -0,0021 |

O excesso cai para quase zero nos modelos comerciais, o que pareceria uma
vitória de viés. **Não é.** Esses modelos quase não aprenderam a marcar ódio
em português (recall PT de 0,01–0,02 no strict), então eles também quase não
marcam as frases de identidade — o excesso baixo é consequência direta de o
modelo não fazer quase nada em PT, não de ele discriminar melhor entre ódio e
menção neutra de identidade. Um modelo que não funciona em português não pode
ser lido como "mais justo" nele. A mesma tabela sob broad confirma o padrão
(excesso cai de 0,1611 para 0,0681/0,0321, acompanhando a mesma queda de
recall PT).

**Teste externo (memotion):** AUC entre 0,49 e 0,52 para todos os candidatos,
nas duas policies — consistente com o já registrado em 14/08 (memotion
sozinho não carrega sinal de ódio via OCR; fica como teste externo, não
confirma nem refuta a comparação comercial×completo).

**Decisão.** Nenhuma variante comercial vai para o registry como candidata a
servir. O corpus comercial de hoje (multioff + vidgen + hatexplain, com ou sem
ToLD-BR) não sustenta um produto bilíngue: o recall em português desaba e a
aparente melhora na sonda de identidade é um artefato da falta de recall, não
um ganho real de equidade. **O que destravaria isso** é o mesmo apontado em
`pt_recall_x_vies.md`: uma fonte em português com licença permissiva e volume
de ódio real, não só texto PT com poucos positivos (ToLD-BR) ou nenhum
(Vidgen/HateXplain são EN). Os 4 modelos ficam versionados em
`models/` como experimento, cada um com a nota explicando o resultado.

Tabelas completas: `reports/tables/commercial_compare_{strict,broad}.csv` e
`reports/tables/commercial_identity_gate_{strict,broad}.csv`. Script:
`scripts/commercial_retrain_compare.py`.

---

## 2026-10-07 — Regra nova: a sonda de identidade não aprova sozinha

**Motivada pelo achado da Etapa 2.2 acima.** Excesso de falso positivo baixo
não é sinal de modelo melhor quando o recall de ódio no mesmo idioma também
caiu. Um modelo que não aprende a marcar ódio em português não marca também
as frases de identidade — isso é ausência de sinal, não ausência de viés.

**Regra adotada:** a sonda de identidade (`hsc bias` /
`scripts/bias_gate_v5.py` / `scripts/commercial_retrain_compare.py`) só
aprova um modelo candidato se o recall de ódio em PT **não cair** em relação
ao modelo servido no momento da comparação. Excesso de sinalização menor com
recall PT menor não conta como melhora — é desqualificado antes mesmo de
olhar o número de excesso. Vale para qualquer candidato futuro (comercial,
sintético, tradução, novo encoder), não só para o experimento acima.

**Verificação do split congelado, pedida depois do incidente de ontem.** Hash
SHA-256 do `corpus_strict.parquet` restaurado recalculado de novo agora:
`9bfda377058d9264339179e02d066b771f9ed722602a3b529087bafa4a3651ae`, idêntico
ao gravado em `data/processed/splits_strict.json` (nunca tocado, de antes do
incidente) e ao conjunto de ids de teste de 8 modelos treinados
independentemente em sessões anteriores (`n=16.261`, mesmo conjunto exato).
Três fontes concordando é a confirmação mais forte possível sem um backup
literal do arquivo.

**Volume de ódio em PT no treino do corpus completo** (o alvo que tradução e
dados sintéticos precisam alcançar, não só igualar):

| policy | treino PT total | ódio PT | % | por fonte |
|---|---|---|---|---|
| strict | 23.743 | 2.039 | 8,6% | pt_fortuna 1.275, hatebr 495, toldbr 269 |
| broad | 23.742 | 6.551 | 27,6% | toldbr 2.875, hatebr 2.401, pt_fortuna 1.275 |

A diferença entre as duas não é o volume de PT (quase igual), é quanto desse
PT conta como ódio: broad promove "offensive" a ódio, o que triplica a
contagem em toldbr e hatebr. Qualquer meta de tradução/síntese para destravar
o recall PT comercial precisa mirar esses números por fonte, não um total
agregado — hatebr e pt_fortuna são justamente as duas fontes de licença
restritiva que o corpus comercial não pode usar.
