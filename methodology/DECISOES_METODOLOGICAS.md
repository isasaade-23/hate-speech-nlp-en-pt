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

**Decisão 1 — Significância pareada por McNemar + correção de Holm.** Comparar modelos
por McNemar exato (binomial) sobre as predições por exemplo, não por diferença de
macro-F1 pontual. As comparações são feitas SÓ dentro de uma política (strict/broad têm
linhas e rótulos de teste diferentes; um teste pareado entre elas não tem sentido). Dentro
da política, todos os modelos usam o mesmo split congelado, então as predições alinham por
`id`. Múltiplas comparações por política são corrigidas por Holm-Bonferroni (α=0,05).

**Motivo.** Diferenças de 1-2 pontos de macro-F1 podem não ser significativas; o revisor
vai exigir o teste. McNemar é o padrão para classificadores pareados no mesmo conjunto.

**Decisão 2 — Predições por exemplo persistidas (reports/predictions/).** train.py grava
(id, y_true, y_score, y_pred) por split; modelos já treinados são reconstruídos do joblib
congelado + corpus congelado, sem re-treinar. É o substrato de McNemar e da calibração.

**Decisão 3 — Calibração reportada (ECE/MCE/Brier + diagrama de confiabilidade).** Scores
passam por min-max antes dos bins para que o decision_function do SVM fique no mesmo eixo
[0,1] das probabilidades (monotônico, não altera ranking/AUC). Uma figura por política.

**Motivo.** O produto expõe um score; calibração diz se ele é interpretável como
probabilidade. É requisito do "Reproducibility/Ethics statement" e da seção de produto.

**Decisão 4 — Trilha SBERT (embeddings densos multilíngues).** Adicionar
`paraphrase-multilingual-MiniLM-L12-v2` (encoder CONGELADO, sem fine-tuning) →
LogReg/LightGBM, como baseline clássico forte que compartilha a ideia multilíngue do
XLM-R sem GPU. O encoder congelado torna o anti-leakage automático (nada é aprendido de
nenhum split); embeddings são cacheados por hash-de-texto (LogReg e LGBM reusam a matriz).

**Motivo.** (a) LightGBM rendia mal e lento sobre TF-IDF esparso de 60k dim; 384 dim
densos são seu habitat natural. (b) TF-IDF de palavra não compartilha vocabulário entre
idiomas — necessário para a transferência cross-lingual.

**Decisão 5 — Experimento de transferência cross-domínio e cross-lingual (destaque).**
Treinar numa fatia e testar em fatia disjunta: cross-domínio EN (tweets↔memes) e
cross-lingual zero-shot (EN→PT e PT→EN). Protocolo idêntico ao principal: features no
treino apenas, limiar ajustado num val da fonte de treino, teste único na fonte-alvo. Como
treino e teste são datasets diferentes, não há leakage na fronteira. Roda com TF-IDF E
SBERT — o contraste (TF-IDF colapsa cross-lingual, SBERT transfere) é o resultado.

**Alternativa rejeitada.** Reportar só números in-distribution. Rejeitada: mede o risco de
domain shift de forma indireta; a transferência explícita é medida direta e vira a Fig. 5.
