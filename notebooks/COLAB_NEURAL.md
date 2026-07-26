# Fase 8 — treino neural no Google Colab (GPU)

Os transformers treinam no Colab (GPU) e devolvem `metrics.json` no MESMO formato do
treino clássico, então `hsc report`/`evaluate` comparam clássico vs neural sem ajustes.

Pré-requisito: o corpus já dividido (`data/processed/corpus_strict.parquet` com a coluna
`split`), gerado localmente por `hsc split`. Suba-o para o Drive ou re-gere no Colab.

## Célula 1 — GPU + repositório
```python
!nvidia-smi -L
from google.colab import drive; drive.mount('/content/drive')
# opção A: clonar o repo (se estiver no GitHub)
# !git clone <URL> /content/hsc && cd /content/hsc
# opção B: já sincronizado no Drive -> ajuste o caminho:
%cd /content/drive/MyDrive/hate-speech-project
```

## Célula 2 — dependências
```python
!pip install -q -r requirements-colab.txt
!pip install -q -e . --no-deps
```

## Célula 3 — treinar um modelo (repita por config e seed)
```python
from hsc.train_neural import train_neural_from_config
corpus = "data/processed/corpus_strict.parquet"   # ou corpus_broad.parquet
for seed in (42, 43, 44):                          # multi-seed p/ IC
    train_neural_from_config(
        "configs/neural/xlmr_multilingual.yaml",
        corpus_path=corpus,
        seed=seed,
    )
```
Repita trocando o config: `bertimbau_pt.yaml` (PT), `bertweet_en.yaml` (EN), e
`distilbert` (gêmeo pequeno p/ o produto). Os pesos vão para `models/<id>/hf` e as
métricas para `reports/metrics/<id>.json`.

## Célula 4 — devolver métricas e comparar
```python
# reports/metrics/*.json já ficam no Drive; localmente rode:
#   hsc report        -> reports/tables/leaderboard.csv (clássico + neural juntos)
```

## Notas
- `fp16=true` exige GPU. Checkpoint frequente (o Colab derruba sessões longas).
- Registre a GPU exata (`nvidia-smi`) no log de experimentos para reprodutibilidade.
- XLM-R multilíngue usa os dois idiomas; BERTimbau/BERTweet filtram por `languages` no config.
