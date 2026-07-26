# Classificador bilíngue de linguagem de ódio (EN / PT)

Detecção binária de discurso de ódio (ódio / não-ódio) em texto de redes sociais, em
inglês e português. O projeto compara **modelos clássicos** (TF-IDF e embeddings sobre
classificadores tabulares) com **transformers** (XLM-R, BERTimbau, BERTweet) sob o mesmo
protocolo, e entrega uma **API de inferência** pronta para produto.

Objetivo duplo: um artefato acadêmico reprodutível (artigo) e um produto implantável.

## Estado

Em construção. Ver o roadmap por fases no plano do projeto e o índice de tarefas.
O log de decisões metodológicas está em `methodology/DECISOES_METODOLOGICAS.md`.

## Dados

Quatro datasets do Kaggle (fonte imutável em `Downloads\hate-speech`):

| id | idioma | domínio | registros | rótulo | licença |
|----|--------|---------|-----------|--------|---------|
| memotion | EN | meme (OCR) | ~6.992 | offensive (4 níveis) | só citação |
| tweets_ip | EN | tweet | ~21.009 | 1/2/3 (inferido) | incerta |
| pt_fortuna | PT | comentário | ~5.670 | binário 0/1 | incerta |
| multioff | EN | meme (OCR) | ~743 | offensive/não | Apache 2.0 |

Detalhes e licenças: `methodology/data_provenance.md`. Mapeamento de rótulos:
`methodology/label_mapping.md` e `configs/labels.yaml`.

## Estrutura

```
configs/      parâmetros de todos os experimentos (a Metodologia é uma leitura dos YAML)
data/         raw (extraídos) / interim / processed / external  (gitignored)
src/hsc/      pacote: ingest, harmonize, clean, langid, splits, features, models, train, evaluate
notebooks/    EDA, probe de rótulos, notebooks de treino no Colab
methodology/  log de decisões, proveniência, mapeamento, limitações  (alimenta o artigo)
reports/      figuras/tabelas/métricas geradas por código
api/ demo/    produto: FastAPI + demo Gradio
deploy/       Docker
tests/        portões de qualidade (anti-leakage, harmonização, contrato da API)
```

## Setup (local, CPU)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"        # núcleo local + ferramentas de dev
pre-commit install
```

O treino neural roda no Google Colab com `requirements-colab.txt` (GPU).

## Pipeline (ordem)

1. `hsc ingest`      extrai e normaliza os 4 datasets ao schema comum (data/interim)
2. Fase 2 probe      resolve os rótulos 1/2/3 do dataset2 (notebook)
3. `hsc harmonize`   aplica configs/labels.yaml, limpa, deduplica (data/processed/corpus.parquet)
4. `hsc split`       split anti-leakage congelado (data/processed/splits.parquet)
5. `hsc langid`      detecção de idioma + avaliação
6. `hsc train -c ...` treina modelos (clássico local; neural no Colab)
7. `hsc evaluate`    comparação, significância, calibração, viés
8. `hsc report`      gera figuras/tabelas do artigo
9. `uvicorn api.main:app`  serve o produto

## Aviso

Ferramenta de pesquisa. As predições são probabilísticas e não substituem moderação
humana. Ver a ethics statement e as limitações em `methodology/`.
