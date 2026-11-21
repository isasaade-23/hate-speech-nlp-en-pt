# API Luciola: termos de uso e privacidade (rascunho, v0 de 21/11/2026)

Rascunho para revisão da Isa e do Vini. Não é parecer jurídico; antes de qualquer uso comercial, advogado.

## Termos de uso
1. **Finalidade.** Acesso de pesquisa e educação, por chave pessoal concedida pela equipe. Não é serviço comercial.
2. **Modelo.** Os pesos são research-only (`LICENSE-MODEL.md`) porque parte do treino vem de bases com licença restrita (HateBR, ToLD-BR). Não revender, não redistribuir os escores em massa.
3. **O que o resultado é.** Probabilidade de um modelo estatístico, não veredito. Recall em português é baixo (0,16 a 0,32 conforme o limiar) e há falsos positivos em frases neutras que citam grupos de identidade. Não usar como única base para moderar, punir, expor ou decidir sobre pessoas; exigir revisão humana.
4. **Chaves.** Pessoais e intransferíveis. A equipe pode revogar a qualquer momento. Limite de 600 unidades por minuto por chave (um texto = uma unidade); acima disso, resposta 429.
5. **Entrada.** Até 2.000 caracteres por texto e 256 textos por lote. Não enviar dados pessoais sensíveis nem conteúdo que a pessoa usuária não tenha base legal para tratar.
6. **Sem garantia.** Serviço experimental, sem SLA, pode sair do ar ou mudar sem aviso. Código AGPL-3.0; quem hospedar uma versão modificada como serviço precisa publicar o código.

## Privacidade (LGPD)
- **O que o servidor guarda:** nome da chave, quantidade de textos e de caracteres, idioma detectado e latência, por requisição. **O texto enviado não é registrado**, nem em erro de validação, e a resposta não o devolve.
- **IP:** o log de acesso do uvicorn está desligado na imagem (`--no-access-log`). Falta conferir/desligar o log de acesso do Caddy no servidor (ou reter por, no máximo, alguns dias).
- **Texto na memória:** existe só durante o processamento da requisição; não vai a disco nem a terceiros.
- **Base legal e papéis:** a equipe é operadora do cálculo; quem envia o texto é a controladora do conteúdo. Contato do encarregado: a definir (Isa).
- **Retenção:** logs de uso por 90 dias (proposta), depois apagados.
- **Direitos do titular:** pedidos pelo contato da equipe; como o texto não é guardado, não há conteúdo a localizar ou apagar.

## Pendências
- Revisão do Vini e de advogado antes de abrir a qualquer pessoa fora da equipe.
- Definir contato (e-mail) e prazo de retenção.
- Publicar em URL própria quando a API sair.
