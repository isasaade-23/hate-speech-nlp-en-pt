# Validação de campo — protocolo e resultados (iniciado 17/08/2026)

Mede como a Luciola se comporta em página real (fora do benchmark), nos dois
modelos entregues: o stack v5 do site (limiar 0,3372) e o linear v5 da extensão
(limiar 0,424). Decide a loja pública, alimenta a seção de domain shift do
paper e dá números reais pra aplicação do Numun.

Regras: nenhum dado colhido entra em treino (só avaliação — não reabre
dedup/split). Os textos colhidos (matérias e comentários de terceiros) NÃO são
commitados no repo público: `reports/field_tests/` fica local (gitignored); só
agregados entram aqui.

## Frente A — Web

### A1. Colheita (17/08/2026)

`scripts/field_harvest.py` replica as regras do `content.js` da extensão
(seletor `p, li, blockquote`; só folhas; 8–2000 chars; corte em 1000; sem o
filtro de visibilidade, que é só do navegador) sobre o HTML servido. Comentários
de YouTube via yt-dlp (API de comentários; mesmos filtros de tamanho).

Material colhido (11 páginas, 1.931 blocos):

| page_id | tipo | blocos |
|---|---|---|
| g1_discord_anpd | notícia política/tec | 90 |
| g1_quaest_sonho / g1_quaest_independentes | notícia eleições 2026 | 56 / 97 |
| folha_dia1_campanha / folha_flavio_copacabana / folha_zema_stf | notícia campanha 2026 | 83 / 87 / 89 |
| g1_agressao_mulher_sinagoga | notícia violência (gênero/religião) | 59 |
| g1_menino_baleado | notícia violência | 50 |
| yt_band_flavio | comentários YouTube (lançamento campanha) | 727 |
| yt_opovo_lula_jingle | comentários YouTube (jingle) | 404 |
| yt_uol_audio_jair | comentários YouTube | 189 |

### A2. Escore duplo (17/08/2026)

`scripts/field_score.py` + `luciola-extension/test/score_file.mjs` (o node usa
exatamente o runtime da extensão; paridade JS×Python já provada em 6,4e-9).

**Resultado preliminar (antes da revisão humana):**

- 34 blocos sinalizados de 1.931 (1,8%).
- Texto de MATÉRIA quase não dispara: 0 flags em 7 das 8 páginas de portal
  (4 flags em g1_quaest_independentes). Jornalismo profissional não é
  confundido com ódio — bom sinal pro caso de uso eleitoral.
- Os flags concentram nos comentários: 30 de 34 (yt_band_flavio 24).
- Concordância entre modelos alta: 27 em ambos, 5 só stack, 2 só linear.
- Fronteira opinião×ódio aparece de cara: hostilidade política genérica
  ("Fora PT.", 0,83) sinalizada junto com desumanização real ("ratos", 0,76) —
  é exatamente o que a revisão humana vai separar.

### A3. Revisão da Isabela (pendente)

`reports/field_tests/revisao_isabela.csv` (364 linhas = 34 sinalizados + 330 de
amostra não-sinalizada, 30/página). Preencher:

- `veredito_isabela`: **TP** (é ódio) · **FP** (não é) · **FN** (da amostra,
  É ódio e passou) · **OK** (da amostra, corretamente limpo)
- `tipo_erro`: identidade_neutra · discurso_reportado · ironia ·
  implicito_perdido · xingamento_sem_alvo · hostilidade_politica · outro
- `obs`: livre

Limitação registrada: anotadora única nesta rodada (sem kappa); 2ª rodada com
segundo anotador antes do paper.

### A4. Sessão manual com a extensão (~30 min, roteiro)

Com a extensão carregada (chrome://extensions → ↻ pra pegar o v5):

1. Abrir 3 das páginas da tabela acima (1 matéria Folha, 1 G1, 1 vídeo YouTube
   com comentários abertos) e clicar Analisar em cada uma.
2. Anotar em `reports/field_tests/sessao_manual.md`: blocos com texto visível
   que NÃO receberam chip (seletor perdeu?); chips quebrando layout; tempo até
   os chips aparecerem (cronometrar a página maior); qualquer bloco > ~1000
   caracteres (transcrição/artigo longo) que ficou sem análise.
3. Testar o slider: baixar o limiar pra 25% numa página de comentários e ver o
   que passa a ser sinalizado (anotar 2–3 exemplos).

### A5. Extensão 0.3.1 (depois de A3/A4)

Segmentação de blocos longos no `content.js` (dividir >1000 chars por sentença
em janelas, chip por trecho), paridade re-testada, CHANGELOG.

## Frente B — Fala transcrita

Desenho em `methodology/inpi/validacao_fala.md` (atualizar referência v4→v5).

### B1. Materiais propostos (aguardando aprovação da Isabela ANTES de baixar)

1. **Debate Band presidencial 2022, 1º turno** (íntegra oficial, 3h52):
   youtube.com/watch?v=WwdgWl_nmKI — multi-candidato, formato clássico.
2. **Debate na Globo 2022, 2º turno Lula×Bolsonaro** (blocos oficiais do g1,
   ~1h32): EK_hxsxWF4I + -woWv61-Urk + MVeRuwkig18 — confronto direto.
3. **Podcast coloquial (~1h dos 2h29)**: "O PRÓXIMO PRESIDENTE DO BRASIL — Kim,
   Arthur do Val e Minatogawa", Inteligência Ltda. #1890
   (youtube.com/watch?v=Rp17MEugiTE) — registro coloquial acalorado,
   multi-falante; contraste com o registro formal dos debates.

### B2–B5 (após OK)

ASR faster-whisper local → segmentação por turno e ~30 palavras
(`scripts/fala_segment.py`) → ~300 segmentos estratificados (100/material)
rotulados pela Isabela com o guia strict → `scripts/fala_eval.py` (ROC-AUC,
PR-AUC, recall-ódio do stack v5 por material) → análise de erro (implícito,
ironia, dog whistle; FP de discurso reportado) → go/no-go do painel de
transcrição (recall < ~0,3 mata). Áudio e transcrições ficam em `data/fala/`
(gitignored).
