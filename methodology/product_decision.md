# Decisão do modelo de produto (Fase 11)

Fonte de verdade: `reports/tables/product_selection_{strict,broad}.csv`, geradas por
`hsc product`. Cruza qualidade (macro-F1, recall de ódio), calibração (ECE), viés de
identidade (gap médio de FPR), latência (p50/p95, medida em CPU sobre textos inéditos) e
tamanho, com a trava de licença.

## Quadro (strict; números-chave)

| modelo | família | F1 | recall ódio | ECE | viés | latência p95 | tamanho | CPU |
|--------|---------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| xlm-roberta | neural | **0,750** | 0,554 | 0,076 | 0,064 | n/a | ~1,1 GB | não* |
| bertimbau | neural | 0,747 | **0,796** | 0,148 | 0,404 | n/a | ~0,4 GB | não* |
| **tfidf_logreg** | clássico | 0,709 | 0,463 | 0,158 | 0,066 | **1,6 ms** | **3,6 MB** | sim |
| tfidf_lgbm | clássico | 0,707 | 0,558 | 0,159 | 0,064 | 7,2 ms | 4,1 MB | sim |
| sbert_lgbm | clássico | 0,683 | 0,520 | **0,032** | 0,110 | 32,6 ms | 3,4 MB+enc | sim |
| tfidf_svm | clássico | 0,672 | 0,497 | 0,027 | 0,069 | 3,9 ms | 4,9 MB | sim |

\* Pesos neurais estão no Colab; não benchmarkados/deployáveis localmente. Precisam de
torch (+GPU para baixa latência).

**Ressalva de tamanho.** O joblib do SBERT (~3,4 MB) NÃO inclui o encoder
`paraphrase-multilingual-MiniLM` (~470 MB) que ele carrega em runtime. Tamanho efetivo do
SBERT em produção ≈ 3,4 MB + 470 MB. O TF-IDF é autossuficiente (só o joblib).

## Recomendação

Três perfis, decisão registrada com os números:

1. **Melhor qualidade (pesquisa / API com GPU):** **XLM-R**. Lidera macro-F1 (0,750 strict)
   e é significativamente melhor que o melhor clássico (McNemar p=0,0026). Custo: torch +
   GPU para latência baixa, ~1,1 GB, e a licença de dados (research-only).

2. **Produto CPU leve, autossuficiente (recomendado para o MVP):** **tfidf_logreg**.
   p95 de 1,6 ms, 3,6 MB, sem encoder externo, F1 a ~4 pontos do XLM-R. Fraquezas:
   pior calibração (ECE 0,16) e, em broad, viés de identidade alto (gap 0,26). Mitigar com
   **temperature scaling** (calibra o score) e mitigação de viés por termo (reamostragem/
   contrafactual). É o candidato de melhor custo-benefício para servir sem GPU.

3. **Se o score calibrado for requisito de produto:** **sbert_lgbm**. Melhor ECE (0,032) e
   viés menor que o TF-IDF em strict, mas 20× mais lento (32 ms) e +470 MB de encoder.

## Trava de licença (bloqueia venda, não o artigo)

**Todos os modelos atuais são research-only**: foram treinados no corpus completo, que
mistura fontes não-comerciais (memotion "só citação", tweets_ip licença incerta,
pt_fortuna "unknown"). A `commercial_whitelist` (labels.yaml) tem só **multioff**
(Apache-2.0, 743 linhas). É pequeno demais para um modelo sozinho.

**Caminho para o modelo comercial:** retreinar a pipeline (mesma config) sobre dados de
licença permissiva apenas: multioff + dados externos a verificar (HateBR/ToLD-BR em PT,
fontes permissivas em EN). Sem isso, o produto fica restrito a pesquisa/demo. Ver
`data_provenance.md`.

## Fronteira de Pareto (CPU-deployáveis)

Nenhum clássico domina todos os eixos: TF-IDF ganha em latência/tamanho, SBERT em
calibração, SVM em calibração+tamanho médio. A escolha é por perfil de uso, não por um
número único. `hsc product` recomputa a fronteira quando as métricas mudam.
