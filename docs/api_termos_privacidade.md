# API Luciola: termos de uso e privacidade

*Última atualização: 2026-11-21. English version below.*

## Termos de uso

1. **Finalidade.** Acesso de pesquisa e educação, por chave pessoal concedida pela equipe. Não é um serviço comercial.
2. **Modelo.** Os pesos são de uso em pesquisa apenas (`LICENSE-MODEL.md`), porque parte do treino vem de bases com licença restrita. Não revender nem redistribuir os escores em massa.
3. **O que o resultado é.** Uma probabilidade de um modelo estatístico, não um veredito. O recall em português é baixo (0,16 a 0,32, conforme o limiar) e há falsos positivos em frases neutras que citam grupos de identidade. Não use como única base para moderar, punir, expor ou decidir sobre pessoas; exija revisão humana.
4. **Chaves.** Pessoais e intransferíveis. A equipe pode revogá-las a qualquer momento. O limite é de 600 unidades por minuto por chave (um texto = uma unidade); acima disso a resposta é 429.
5. **Entrada.** Até 2.000 caracteres por texto e 256 textos por lote. Não envie dados pessoais sensíveis nem conteúdo que você não tenha base legal para tratar.
6. **Sem garantia.** Serviço experimental, sem SLA; pode sair do ar ou mudar sem aviso. O código é AGPL-3.0: quem hospedar uma versão modificada como serviço precisa publicar o código.

## Privacidade (LGPD)

- **O que o servidor registra:** por requisição, o nome da chave, a quantidade de textos e de caracteres, o idioma detectado e a latência. **O texto enviado não é registrado**, nem em erro de validação, e a resposta não o devolve.
- **IP:** não é registrado. O log de acesso do servidor web e do uvicorn está desligado.
- **Texto:** existe na memória apenas durante a requisição; não vai a disco nem a terceiros.
- **Retenção:** esses registros de uso ficam no log do contêiner, que é rotacionado por tamanho (no máximo 30 MB); não há outro armazenamento.
- **Papéis:** a equipe opera o cálculo; quem envia o texto controla o conteúdo.
- **Direitos do titular e contato:** como o texto não é guardado, não há conteúdo a localizar ou apagar. Dúvidas e pedidos: https://github.com/isasaade-23/hate-speech-nlp-en-pt/issues

---

## English

**Terms.** Research and education access by personal key granted by the team; not a commercial service. Model weights are research-only (`LICENSE-MODEL.md`). The output is a probability from a statistical model, not a verdict; Portuguese recall is low (0.16 to 0.32 depending on threshold) and neutral sentences mentioning identity groups can be flagged. Do not use it as the sole basis to moderate, punish or expose anyone; require human review. Keys are personal and revocable; limit 600 units per minute per key (one text = one unit; 429 above it). Input: up to 2,000 characters per text, 256 texts per batch; do not send sensitive personal data. Experimental service, no SLA. Code is AGPL-3.0.

**Privacy.** Per request the server records only the key name, number of texts and characters, detected language and latency. The submitted text is not logged and is not echoed back; it stays in memory only while the request is processed. IP addresses are not logged. The usage records live in the container log, rotated by size (30 MB at most). Contact: https://github.com/isasaade-23/hate-speech-nlp-en-pt/issues
