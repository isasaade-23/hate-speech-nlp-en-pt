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

## Ético
- Uso dual: um classificador de ódio pode ser mal usado. Incluir ethics statement, avisos
  de conteúdo e a ressalva de que a predição é probabilística e não substitui moderação
  humana.
