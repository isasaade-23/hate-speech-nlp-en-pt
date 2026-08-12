# Mapeamento de rótulos para binário (ódio / não-ódio)

Versão legível de `configs/labels.yaml`. Alvo: `1 = ódio`, `0 = não-ódio`.
Duas políticas: `strict` (padrão) e `broad`.

## Tabela

| Fonte | Rótulo original | strict | broad | Confiança | Justificativa |
|-------|-----------------|:------:|:-----:|-----------|---------------|
| memotion | hateful_offensive | 1 | 1 | alta | nível explícito "hateful" |
| memotion | very_offensive | 0 | 1 | baixa | ofensivo, não necessariamente ódio |
| memotion | slight | 0 | 0 | alta | não é ódio |
| memotion | not_offensive | 0 | 0 | alta | não é ódio |
| tweets_ip | 1 | 1 | 1 | média | classe com slurs/epítetos (provisório, ver Fase 2) |
| tweets_ip | 2 | 0 | 1 | baixa | hostil mas não claramente ódio |
| tweets_ip | 3 | 0 | 0 | média | majoritariamente benigno/neutro |
| pt_fortuna | hatespeech_comb = 1 | 1 | 1 | alta | rótulo binário de ódio feito sob medida (padrão-ouro) |
| pt_fortuna | hatespeech_comb = 0 | 0 | 0 | alta | não é ódio |
| multioff | offensive | (fora) | 1 | baixa | ofensividade, não ódio; sob strict fica fora do primário |
| multioff | Non-offensiv (truncado) | 0 | 0 | alta | casar por prefixo, não igualdade |
| hatebr | hate (código `hate_speech` 1–9) | 1 | 1 | alta | ódio anotado por especialista (as 9 categorias) |
| hatebr | offensive_nothate (`hate_speech` = −1) | 0 | 1 | alta | ofensivo mas não-ódio na anotação (bucket próprio) |
| hatebr | neither (`hate_speech` = 0) | 0 | 0 | alta | não-ofensivo |

## Notas
- **strict vs broad.** strict só marca 1 quando há ódio explícito. broad dobra
  ofensividade em ódio. A diferença entre as duas é um resultado de destaque do artigo.
- **multioff sob strict.** Como MultiOFF rotula ofensividade e não ódio, sob strict não
  há exemplo de ódio confirmado; a fonte fica fora do corpus primário (só negativos
  enviesariam) e é mantida como conjunto auxiliar/teste em inglês. Sob broad, offensive
  vira 1 e a fonte entra no primário.
- **tweets_ip provisório.** O mapeamento 1/2/3 é inferido. A Fase 2 (probe) confirma,
  restringe à fronteira ódio-vs-resto, ou rebaixa a fonte. Ver DECISOES_METODOLOGICAS.md.
- **hatebr nativo.** Ao contrário do tweets_ip, a fronteira ofensa≠ódio do HateBR é
  anotada, não inferida: o loader deriva as 3 vias do código `hate_speech` (`0`→neither,
  `-1`→offensive_nothate, 1–9→hate). broad reproduz o `offensive_language` original; strict
  isola os 702 casos de ódio real. Fonte research-only (Sinch), fora do produto.
- **label_confidence.** Cada linha do corpus carrega `label_confidence` (high/low). Linhas
  low (ofensivo tratado como proxy de ódio) permitem ablação e análise de robustez.
