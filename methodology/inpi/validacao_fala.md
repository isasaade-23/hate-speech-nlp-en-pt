# Validação em fala transcrita — desenho do experimento (anexo do paper)

Pergunta da Isabela (14/08/2026): o modelo, treinado em texto escrito de rede
social, funciona em fala transcrita (debates, lives)? Resposta honesta: não dá
pra afirmar sem medir, e a expectativa fundamentada é de queda.

## Por que esperar queda

1. Transferência entre domínios já medida no estudo: tweets→memes caiu a
   macro-F1 ~0,50 (quase aleatório). Fala é outro domínio.
2. Os sinais mais fortes do modelo são de SUPERFÍCIE escrita: grafias
   propositais de slur, char n-grams de ofuscação, emojis demojizados. ASR não
   produz nada disso: normaliza grafia e não tem emoji.
3. Registro: fala pública (debate) quase não tem slur explícito; o ódio ali é
   implícito e codificado, exatamente o maior balde de falso negativo do estudo.

## Desenho

1. **Material**: 2 debates eleitorais antigos (íntegras públicas no YouTube) +
   1 hora de live/podcast com linguagem coloquial (contraste de registro).
2. **ASR**: faster-whisper local (CPU) ou Colab; guardar transcrição bruta e
   com pontuação; segmentar por turno de fala e por janela de ~30 palavras.
3. **Rotulagem**: amostra estratificada de ~300 segmentos (100 por material),
   2 anotadores independentes com o MESMO guia strict/broad do corpus; reportar
   kappa. Segmentos com discordância vão a desempate.
4. **Métricas**: ROC-AUC, PR-AUC e recall-ódio do stack v4 sobre os segmentos,
   por material e por política; comparar com o teste v4 como referência.
5. **Análise de erro**: catalogar falsos negativos por tipo (implícito, ironia,
   dog whistle) e falsos positivos (citação de fala de terceiro, discurso
   reportado — armadilha clássica de debate: candidato CITANDO ódio do outro).
6. **Produto do experimento**: seção "domain shift pra fala" no paper + decisão
   go/no-go sobre o painel do debate (se recall-ódio < ~0,3, o painel de
   transcrição não sai; o termômetro de CHAT ao vivo segue viável porque chat é
   nosso domínio de treino).

## Custo estimado

Local e barato: ASR em CPU roda de um dia pro outro; rotulagem ~300 segmentos é
uma tarde de trabalho a dois. Nenhum dado novo entra no corpus de treino (só
avaliação), então NÃO reabre dedup/split.
