# Fase 8 — treino neural no Google Colab (GPU)

Fluxo turnkey em 3 movimentos. Os transformers treinam no Colab (GPU) e devolvem
`metrics.json` **e** predições por-exemplo no MESMO formato do treino clássico, com o
MESMO ajuste de limiar na validação — então `hsc report`/`analyze`/`bias` comparam
clássico vs neural sem ajuste nenhum.

O notebook pronto é `notebooks/colab_neural.ipynb` (abra no Colab e rode em ordem). Este
`.md` é o resumo do fluxo.

## 1 · Empacotar (local)
```
.venv/Scripts/python.exe notebooks/make_colab_bundle.py     # -> colab_bundle.zip (~11 MB)
```
Inclui o corpus congelado (UTF-8 corrigido), o pacote `hsc`, os configs e
`requirements-colab.txt`. Exclui dados brutos e o cache de embeddings. Suba
`colab_bundle.zip` para a raiz do seu Google Drive.

## 2 · Treinar (Colab)
Abra `notebooks/colab_neural.ipynb`, escolha **GPU (T4)** e rode as células:
desempacota o bundle → instala → **trava anti-mojibake** (confere UTF-8 do PT) → treina a
matriz `3 configs × {strict, broad} × seeds`. O laço é resumível (pula o que já terminou;
o Colab derruba sessões longas). Comece com `SEEDS = [42]`; use `[42, 43, 44]` para IC.

- `xlmr_multilingual.yaml` — XLM-R base, um modelo para EN+PT (o central da comparação).
- `bertimbau_pt.yaml` — BERTimbau, filtra `languages: [pt]`.
- `bertweet_en.yaml` — BERTweet, filtra `languages: [en]`.

A última célula empacota `reports/metrics/*.json` + `reports/predictions/*.parquet` +
o registry neural em `hsc_neural_results.zip` no Drive.

## 3 · Fechar a comparação (local)
Baixe `hsc_neural_results.zip` para a raiz do repo e:
```
.venv/Scripts/python.exe notebooks/merge_neural_results.py hsc_neural_results.zip
hsc report      # leaderboard clássico + neural juntos
hsc analyze     # McNemar + calibração incluindo os neurais
hsc bias        # viés de identidade dos neurais
```
O merge funde as entradas neurais no `models/registry.json` local sem tocar nas
clássicas.

## Notas
- `fp16=true` exige GPU. Registre a GPU exata (`nvidia-smi`) no experiment_log.
- Limiar ajustado na validação em ambos (clássico e neural) — comparação justa.
- A análise de erro (Fase 9) já aponta ONDE os transformers precisam ganhar: **ódio
  implícito** (206/313 dos falso-negativos do melhor clássico não têm palavrão).
