# Model card — pt_logreg_strict_s42

**EXPERIMENTO, NÃO SERVIDO.**

- Config: pt_logreg  |  Política: strict  |  Semente: 42  |  git: n/d
- Corpus de treino: `corpus_strict.parquet` (versão v1; não é o strict v2)
- sha256 de `model.joblib`: `35d068ef4f8e517450caaf4f3122ae621b843f598ab3ca1f6590f22f7b86166c`
- Limiar: 0.5666  |  macro-F1 teste: 0.7572  |  ROC-AUC teste: 0.9015

Logreg só português. EXPERIMENTO, NAO SERVIDO: dobra o recall de odio em PT (0.3162 -> 0.6225) mas triplica o falso positivo em frases neutras com termo de identidade (9/20 contra 3/20 do stack), marcando pessoas LGBT, negras e muculmanas falando de si. Ver methodology/pt_recall_x_vies.md

## Uso e limites
Classificador de pesquisa para discurso de ódio em redes sociais EN/PT. Probabilístico; não é oráculo de moderação. Os pesos são research-only, pois o treino inclui bases de licença restrita (HateBR, ToLD-BR): ver `LICENSE-MODEL.md` e `methodology/data_provenance.md`. Limites e viés: `methodology/limitations.md`.
