# Proveniência e licenças dos dados

Fonte original (imutável): `C:\Users\Renato\Downloads\hate-speech`. Os zips não são
editados. data/raw guarda apenas os arquivos tabulares extraídos (ver decisão de 2026-07-26).

Hashes SHA-256 dos zips de origem: preenchidos por `src/hsc/ingest` na primeira execução
(gravados em data/raw/PROVENANCE.json). Placeholder abaixo até rodar.

| id | dataset | idioma | domínio | zip | membro tabular | registros | codificação |
|----|---------|--------|---------|-----|----------------|-----------|-------------|
| memotion | Memotion Dataset 7k | EN | meme (OCR) | dataset1.zip | memotion_dataset_7k/labels.csv | ~6.992 | utf-8-sig |
| tweets_ip | Hate and offensive speech detection | EN | tweet | dataset2.zip | Hate and offensive speech detection.csv | ~21.009 | utf-8 |
| pt_fortuna | Portuguese Hate Speech (Fortuna 2019) | PT | comentário web | dataset3.zip | 2019-05-28_portuguese_hate_speech_binary_classification.csv | ~5.670 | utf-8 (corrigido 27/07; era lido como latin-1) |
| multioff | MultiOFF / Hate Speech Detection Dataset | EN | meme (OCR) | dataset4.zip | Dataset/Split Dataset/{Training,Validation,Testing}_meme_dataset.csv | 445/149/149 | utf-8-sig |
| hatebr | HateBR (Vargas et al. 2022) | PT | comentário web (Instagram) | hatebr.zip | HateBR.csv | 7.000 | utf-8 |
| toldbr | ToLD-Br (Leite et al. 2020) | PT | tweet | toldbr.zip | ToLD-BR.csv | 21.000 | utf-8 (109 U+FFFD nativos) |

**Beta 2.0 (14/08/2026):** `memotion` rebaixado a conjunto auxiliar/teste externo
(`include_in_primary: false` nas duas políticas). Motivo: AUC por fonte de 0.547 no teste
v3 (aleatório) porque o texto OCR sozinho não carrega o ódio do meme; o alvo de deploy é
texto de rede social. O dataset segue ingerido e recuperável.

## Licenças (crítico para o produto comercial)

| id | licença declarada no Kaggle | uso em pesquisa | uso comercial |
|----|-----------------------------|-----------------|----------------|
| memotion | "Other" (só citação): "allowed to be used in any paper, only upon citation" | sim, com citação | **incerto** (não é licença comercial) |
| tweets_ip | não recuperável localmente (sem página salva) | verificar na origem | **incerto** |
| pt_fortuna | "Unknown" no Kaggle (corpus Fortuna et al. 2019) | sim (uso acadêmico usual) | **incerto**, verificar com autores |
| multioff | **Apache 2.0** | sim | **sim** (permissiva) |
| hatebr | **research-only** ("strictly for academic and research purposes... commercial use expressly prohibited without prior written consent of Sinch") | sim (uso acadêmico) | **não** (comercial vedado sem consentimento) |
| toldbr | **CC BY-SA 4.0** (dados; código MIT) | sim, com atribuição | **cinza**: share-alike sobre pesos derivados é juridicamente não testado; mantido fora da whitelist |

### Regra do produto (dois modelos)
- **Modelo de pesquisa:** treinado em todos os datasets. Uso research-only. Não
  redistribuir comercialmente sem esclarecer licenças de memotion, tweets_ip e pt_fortuna.
- **Modelo de produto:** treinado apenas na `commercial_whitelist` de configs/labels.yaml
  (hoje: multioff) mais dados externos de licença permissiva a verificar (candidatos PT:
  HateBR, ToLD-BR, OffComBR; candidatos EN: OLID/OffensEval, HatEval sob suas licenças).

### Pendências de licenciamento (bloqueiam venda, não o artigo)
1. Confirmar a licença de tweets_ip na página original do Kaggle.
2. Contatar autores de pt_fortuna sobre uso comercial, ou substituir por HateBR/ToLD-BR
   com licença clara.
3. Se memotion/pt_fortuna/tweets_ip não liberarem uso comercial, o produto vende só o
   modelo treinado na whitelist + dados externos permissivos.

## Citações a incluir no artigo
- Memotion: Sharma et al., SemEval-2020 Task 8 (Memotion Analysis).
- pt_fortuna: Fortuna et al., 2019 (Portuguese hate speech dataset).
- multioff: Suryawanshi et al., 2020 (MultiOFF / offensive memes).
- tweets_ip: confirmar autoria/citação na origem.
- hatebr: Vargas et al., 2022 (HateBR: expert-annotated Brazilian Instagram hate speech corpus),
  distribuído pela Sinch; fonte `franciellevargas/HateBR`. Wrapper HF: `ruanchaves/hatebr`.
