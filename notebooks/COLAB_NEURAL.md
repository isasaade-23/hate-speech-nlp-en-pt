# Fase 8 — treino neural no Google Colab (GPU)

Notebook autossuficiente: `notebooks/colab_neural.ipynb`. Não precisa clonar o repo nem
instalar o pacote — o código de treino e de métricas está embutido no notebook, idêntico
ao do pipeline local (verificado: saída byte-a-byte igual à de `hsc.evaluate`). Assim os
transformers entram direto no leaderboard/McNemar/calibração locais, com a MESMA regra de
limiar na validação — comparação clássico-vs-neural justa por construção.

## Fluxo (2 movimentos)

**1. No Colab.** Abra `notebooks/colab_neural.ipynb`, escolha **GPU (T4)** e rode as
células em ordem:
- célula 3: sobe `corpus_strict.parquet` e `corpus_broad.parquet` (de `data/processed/`
  no seu PC) pelo widget de upload, com uma **trava anti-mojibake** que recusa PT
  corrompido (garante o UTF-8 corrigido);
- células 4-7: treinam `xlmr_multilingual`, `bertimbau_pt`, `bertweet_en` × `{strict,
  broad}` × seeds. Laço **resumível** (o Colab derruba sessões longas; rode de novo e ele
  pula o que já terminou). Comece com `SEEDS = [42]`; use `[42, 43, 44]` para IC;
- célula 8: baixa `hsc_neural_results.zip` (métricas + predições + registry neural).

**2. No local.** Baixe `hsc_neural_results.zip` para a raiz do repo e:
```
.venv/Scripts/python.exe notebooks/merge_neural_results.py hsc_neural_results.zip
hsc report      # leaderboard clássico + neural juntos
hsc analyze     # McNemar + calibração incluindo os neurais
hsc bias        # viés de identidade dos neurais também
```
O merge funde as entradas neurais no `models/registry.json` local sem tocar nas clássicas.

## Configs (inline no notebook, espelham configs/neural/*.yaml)
- **xlm-roberta-base** — um modelo para EN+PT (o central da comparação).
- **bert-base-portuguese-cased (BERTimbau)** — filtra PT.
- **bertweet-base** — filtra EN.

## Notas
- `fp16=true` exige GPU. Registre a GPU exata (`nvidia-smi`) no experiment_log.
- A análise de erro (Fase 9) já aponta ONDE os transformers precisam ganhar: **ódio
  implícito** (206/313 dos falso-negativos do melhor clássico não têm palavrão).
- `src/hsc/train_neural.py` roda a MESMA lógica caso um dia se prefira treinar via CLI
  local com GPU; o notebook é a via recomendada para o Colab.
