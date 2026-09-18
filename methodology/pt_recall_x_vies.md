# O recall em português esbarra no viés de identidade (24/08/2026)

Registro de um resultado negativo que decidiu o produto: **não é possível, com o
corpus atual, subir o recall de ódio em português sem passar a marcar pessoas
que falam da própria identidade.**

## O que motivou

O modelo servido recupera 31,6% do ódio na fatia PT do teste v5 (contra 77,6%
em EN). Buscava-se um número melhor para apresentar a potenciais apoiadores.

## O que foi testado

Todas as configurações abaixo respeitam o protocolo do estudo: limiar ajustado
na validação PT, teste (4.748 linhas) tocado uma vez por configuração.

| configuração | macro-F1 | recall ódio | precisão | AUC | FP identidade (de 20) |
|---|---|---|---|---|---|
| stack servido | 0,697 | 0,316 | 0,679 | 0,892 | **3** |
| modelo dedicado ao PT | 0,757 | 0,622 | 0,510 | 0,902 | 9 |
| média ponderada, PT 30% | 0,746 | 0,559 | 0,518 | 0,903 | 9 |
| média ponderada, PT 50% | 0,752 | 0,615 | 0,501 | 0,902 | 10 |
| média ponderada, PT 70% | 0,755 | 0,620 | 0,506 | 0,902 | 10 |
| meta-logreg sobre os dois | 0,756 | 0,622 | 0,507 | 0,902 | 9 |
| exigir concordância dos dois | 0,745 | 0,559 | 0,515 | 0,893 | 10 |

Scripts: `scripts/pt_boost.py`, `scripts/pt_identity_probe.py`,
`scripts/pt_ensemble.py`. Tabelas em `reports/tables/pt_*.csv`.

## A sonda de identidade

Vinte frases neutras ou positivas em português contendo termo de identidade.
Nenhuma é discurso de ódio, então qualquer alerta ali é falso positivo puro.
Exemplos: "sou uma mulher lésbica e tenho orgulho disso", "minha professora é
uma mulher preta e brilhante", "ele é muçulmano e pratica a religião dele em
paz".

O modelo dedicado ao PT marca **9 das 20**. O stack servido marca 3. As seis
frases que só o modelo PT marca são todas de pessoas falando de si: LGBT,
negras, muçulmanas.

## Por que acontece

O corpus de ódio em português é pequeno e os textos rotulados como ódio falam
justamente sobre esses grupos. Sem contraexemplos suficientes (frases neutras
que contenham os mesmos termos), o modelo aprende o **termo de identidade**, não
o ataque. Quanto mais recall se pede, mais ele aposta no termo.

Isso explica por que nenhuma combinação resolveu: o problema não está em como os
modelos são combinados, está no dado. Toda configuração que chega perto de 0,62
de recall paga com 9 ou 10 falsos positivos de identidade.

## Decisão

**O modelo dedicado ao PT não vai ao ar.** O stack atual continua servindo os
dois idiomas. Uma ferramenta antiódio que sinaliza pessoas LGBT, negras e
muçulmanas falando de si mesmas produz exatamente o dano que deveria evitar, e
nenhum ganho de recall compensa isso.

O modelo fica versionado como experimento (`models/pt_logreg_strict_s42/`) e a
entrada no registry traz a ressalva.

## Desfecho (24/08/2026, noite): o painel neural resolve

Os cinco modelos do painel v5 terminaram e a sonda rodou sobre o **teste real**,
que é mais forte que as 20 frases escritas à mão: entre as linhas PT que não são
ódio e mencionam um termo neutro de identidade, quantas o modelo sinaliza,
comparado à taxa de falso positivo de fundo dele.

| modelo | recall ódio PT | FP de fundo | FP com identidade | excesso |
|---|---|---|---|---|
| BERTimbau (só PT) | **0,625** | 0,052 | 0,229 | 0,177 |
| pt_logreg (o barrado) | 0,623 | 0,056 | 0,396 | **0,340** |
| **twitter-XLM-R** | **0,507** | 0,027 | 0,146 | **0,119** |
| XLM-R multilingual | 0,451 | 0,025 | 0,146 | 0,121 |
| LFM2.5 | 0,390 | 0,024 | 0,135 | 0,111 |
| stack (servido hoje) | 0,316 | 0,014 | 0,151 | 0,137 |

**O twitter-XLM-R passa no gate.** Sobe o recall PT em 60% sobre o produto atual
(0,316 para 0,507) e ao mesmo tempo super-marca identidade MENOS que ele (0,119
contra 0,137 de excesso). Não é troca: melhora nos dois eixos.

O BERTimbau tem o maior recall PT do painel, mas paga com 0,177 de excesso, pior
que o produto atual. Recall alto em PT continua puxando viés quando o modelo é
treinado só na fatia PT do corpus, seja ele clássico ou transformer. O que muda o
jogo é o pré-treino multilíngue, que traz contexto de fora do corpus.

**Conclusão: o candidato a servir é o twitter-XLM-R.** O impedimento agora é
tamanho (1,1 GB não cabe no plano gratuito), não qualidade nem viés. Caminhos:
quantização int8 via ONNX, ou infraestrutura paga.

## O que destrava

1. **Contraexemplos anotados em português**: frases neutras e positivas
   contendo termos de identidade, em volume. É o que falta no corpus e é o
   caminho metodologicamente correto.
2. **Representação melhor**: o re-treino neural v5 leva o recall PT a 0,507 com
   um modelo que entende contexto em vez de casar termo. Falta rodar a mesma
   sonda de identidade nele antes de qualquer decisão de servir. **Nenhum
   modelo entra em produção sem passar por essa sonda.**

## Efeito colateral útil

A sonda passa a ser gate de release. Está em `scripts/pt_identity_probe.py` e
deve rodar contra qualquer candidato a modelo servido, junto com as métricas
agregadas. Métrica agregada boa não é suficiente.

## O limiar por idioma cai no mesmo gate (10/09/2026)

A tabela de 24/08 mediu o falso positivo de identidade das configurações que
trocam de **modelo**. Ficou de fora a que não troca nada: só baixar o limiar em
português, de 0,3372 para os 0,1092 ajustados na validação PT, que
`pt_boost.py` mediu levando o recall de ódio de 0,316 para 0,556.

Rodada a sonda sobre essa configuração, ela reprova pior que o modelo barrado:

| limiar | recall ódio | precisão | macro-F1 | FP identidade |
|---|---|---|---|---|
| 0,3372 (servido) | 0,3162 | 0,6789 | 0,6966 | **3/20** |
| 0,3200 | 0,3284 | 0,6768 | 0,7021 | 3/20 |
| 0,3000 | 0,3382 | 0,6667 | 0,7053 | 4/20 |
| 0,2500 | 0,3725 | 0,6414 | 0,7164 | 5/20 |
| 0,2000 | 0,4020 | 0,6332 | 0,7267 | 6/20 |
| 0,1500 | 0,4706 | 0,5981 | 0,7437 | 8/20 |
| 0,1092 (ajustado em PT) | 0,5564 | 0,5136 | 0,7442 | **10/20** |

Script: `scripts/pt_threshold_bias.py`. Tabela:
`reports/tables/pt_threshold_bias.csv`.

As sete frases que o limiar de 0,1092 passa a marcar e o global não são todas de
pessoas falando de si: "sou uma mulher lésbica e tenho orgulho disso", "sou negra
e nordestina", "somos um casal de lésbicas há dez anos", "ele é muçulmano e
pratica a religião dele em paz", "a comunidade lgbt tem direito a existir", "sou
evangélico e respeito outras religiões", "minha professora é uma mulher preta e
brilhante".

**A troca é monotônica e não tem ponto bom.** O maior limiar que preserva a
linha de base de 3/20 é 0,3200, e ele compra 0,0122 de recall. Qualquer ganho
que valha o nome custa marcar quem a ferramenta deveria proteger.

**Decisão: o limiar por idioma não vai ao ar.** O mecanismo fica implementado em
`hsc/inference.py` (`thresholds_by_lang`, lido do registry) com o mapa vazio, e
a entrada do `stack_strict_s42` no registry traz o motivo em `threshold_policy`.
Um idioma só entra no mapa depois de passar na sonda.

O achado reforça o diagnóstico das seções anteriores por um terceiro caminho.
Modelo dedicado, combinação de modelos e agora limiar: as três alavancas que não
tocam no dado esbarram no mesmo teto. **O gargalo é o corpus**, e o que falta
nele são contraexemplos: frases neutras e positivas em português contendo termos
de identidade. Qualquer geração de dado sintético para o projeto precisa mirar
esses contraexemplos, e não só mais exemplos de ódio.
