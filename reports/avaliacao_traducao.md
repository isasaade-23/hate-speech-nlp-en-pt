# Qualidade das traduções do piloto, 2026-10-14

Comparação pareada nos mesmos ids, sem texto, só ids. Defeitos e máscaras definidos em `scripts/avaliar_traducao.py`. Máscara (palavrão com letra trocada) é contada à parte e não é defeito.

| par | motor | n | html | codificacao | alfabeto | ingles | vazio | igual | tamanho | quebrada | máscaras |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MarianMT | Gemini | 248 | 0.0% | 0.0% | 0.4% | 0.8% | 0.4% | 0.0% | 0.0% | 1.6% | 2 em 2 itens |
| MarianMT | MarianMT | 248 | 0.0% | 0.0% | 0.0% | 4.8% | 0.0% | 0.0% | 0.0% | 4.8% | 1 em 1 itens |
| Qwen | Gemini | 252 | 0.0% | 0.0% | 0.0% | 2.4% | 0.0% | 0.0% | 0.0% | 2.4% | 1 em 1 itens |
| Qwen | Qwen | 252 | 2.4% | 0.0% | 1.6% | 10.3% | 0.0% | 0.0% | 0.4% | 14.7% | 3 em 2 itens |
| todos | Gemini | 500 | 0.0% | 0.0% | 0.2% | 1.6% | 0.2% | 0.0% | 0.0% | 2.0% | 3 em 3 itens |

Status do Gemini nos itens avaliados: ok 499, bloqueado 1

## Exemplos por defeito (até 5 ids)

- **MarianMT**: ingles: hatexplain_9109, hatexplain_6074, vidgen_19708, hatexplain_15195, hatexplain_7733; mascara: vidgen_5958
- **Qwen**: html: hatexplain_14872, hatexplain_17285, hatexplain_9374, hatexplain_6562, hatexplain_13347; alfabeto: hatexplain_2145, vidgen_2072, vidgen_2938, vidgen_30885; ingles: hatexplain_1294, vidgen_40013, vidgen_35309, vidgen_28797, hatexplain_9941; tamanho: hatexplain_4841; mascara: vidgen_12093, vidgen_26136
- **Gemini**: alfabeto: hatexplain_16226; ingles: hatexplain_10527, hatexplain_15195, vidgen_26136, hatexplain_7573, vidgen_26090; vazio: vidgen_29130; mascara: vidgen_7014, vidgen_5958, vidgen_8491

## Palavras em inglês que mais sobraram

- **Gemini**: for (2), the (2), right (1), of (1), dick (1), dyke (1), country (1)
- **MarianMT**: hoe (4), kike (2), shawty (1), faggot (1), right (1), nigga (1), chink (1), for (1), thot (1), spic (1)
- **Qwen**: kike (4), out (3), dyke (3), nigger (2), black (2), for (2), to (2), fag (1), totally (1), got (1)
