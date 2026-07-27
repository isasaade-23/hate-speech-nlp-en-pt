# Limitações (acumulando para a Discussão)

Lista viva. Cada item vira uma frase ou parágrafo na seção de Limitações do artigo.

## Dados e rótulos
- **Ofensa não é ódio.** memotion very_offensive, tweets_ip classe 2 e multioff offensive
  medem ofensividade, não ódio. Tratados via políticas strict/broad e label_confidence.
- **Rótulos do dataset2 inferidos.** O significado 1/2/3 não é documentado; mapeamento
  sujeito ao resultado da Fase 2.
- **Português pequeno e de fonte única.** pt_fortuna tem cerca de 5.670 exemplos, só
  comentários. Limita o modelo PT dedicado e a generalização.

## Confusões e viés
- **Idioma confundido com domínio.** PT = comentário; EN = tweet + meme. Conclusões por
  idioma precisam da quebra por source_dataset e do experimento de transfer.
- **OCR de meme é outra modalidade.** Texto curto, ruidoso, às vezes dependente da imagem.
  Text-only tende a render menos nos memes.
- **Tema do dataset2.** Fortemente Israel/Palestina; risco de viés por termo de identidade.
  Sondagem de viés usando as 77 categorias do pt_fortuna.

## Método e produto
- **Determinismo em GPU.** Treino de transformers não é bit-exato; reportar variância
  multi-seed com intervalos de confiança.
- **Erro de detecção de idioma** entra no erro ponta a ponta; medido separadamente.
- **Licenciamento** restringe o modelo comercializável (ver data_provenance.md).

## Achados da análise de erro (Fase 9)
- **Ódio implícito é o maior ponto cego.** No melhor modelo strict, 206 dos 313 falso-
  negativos são ódio sem palavrão/slur. Modelos de superfície (TF-IDF) dependem de termos
  explícitos; ódio implícito/sarcástico escapa. Transformers devem ajudar aqui (Fase 8).
- **Over-flag por slur.** Parte dos falso-positivos são textos não-ódio que contêm
  palavrão (reapropriação, citação, humor). O modelo super-confia na presença do termo.
- **Super-marcação do PT.** O falso-positivo em PT (0,185) é ~7x o do EN (0,027): fonte
  menor e fronteira ofensa/ódio mais densa levam o modelo a chamar PT de ódio em excesso.

## Viés por termo de identidade (Fase 9)
- **Viés não-intencional confirmado.** Em textos NÃO-ódio, citar grupos de identidade
  infla o falso-positivo: orientação sexual chega a FPR 0,75 vs. 0,17 de fundo (gap
  +0,59); religião, nacionalidade/imigração e gênero também. Medido com termos neutros
  (não slurs), bilíngue. Padrão Dixon et al. 2018. Entra no Ethics statement; mitigação
  futura: reamostragem/contrafactual por termo, ou penalização de viés.
- SBERT tende a gaps um pouco menores que TF-IDF, mas o viés persiste em todos os modelos.

## Encoding e proveniência (lição da Fase 9)
- **Bug de encoding do PT corrigido.** O pt_fortuna foi lido como latin-1 quando é UTF-8,
  corrompendo todos os acentos; detectado por auditoria de bytes (C3 A9 = 'é' em UTF-8),
  não por glifo de console. Regra incorporada ao protocolo: **validar encoding por
  codepoint/byte**, nunca pela aparência no terminal. Impacto maior era no SBERT/cross-
  lingual; corrigido e recomputado (ver DECISOES_METODOLOGICAS 2026-07-27).

## Ético
- Uso dual: um classificador de ódio pode ser mal usado. Incluir ethics statement, avisos
  de conteúdo e a ressalva de que a predição é probabilística e não substitui moderação
  humana.
