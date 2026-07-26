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
