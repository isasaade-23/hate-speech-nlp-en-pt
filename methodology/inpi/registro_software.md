# Registro de Programa de Computador (INPI e-Software) — dossiê (14/08/2026)

Registro de AUTORIA (direito autoral de software, Lei 9.609/98). Não é patente e
não conflita com código aberto: o código continua livre no GitHub (AGPL-3.0 desde
24/08/2026; era MIT quando o dossiê foi montado); o registro prova
autoria e data, o que interessa pra tese, transferência de tecnologia e disputas.
Validade: 50 anos. Protocolo é da Isabela (gov.br + GRU).

## Custo e prazo

- GRU código **730** (pedido eletrônico via e-Software): **R$ 210** (conferido
  14/08/2026; confirmar na tabela do INPI).
- Concessão típica em ~10 dias úteis (é registro declaratório por hash, sem exame
  de mérito).

## O que registrar

Um pedido pode cobrir o programa "Luciola" como obra única. Recomendação: um
registro cobrindo os dois repositórios (pesquisa + demo) e a extensão, com o hash
de um pacote único:

```powershell
# gerar o pacote e o hash SHA-512 (rodar na pasta C:\Users\Renato)
Compress-Archive -Path hate-speech-project, hate-speech-space, luciola-extension -DestinationPath luciola_codigo_v1.zip
Get-FileHash luciola_codigo_v1.zip -Algorithm SHA512
```

Guardar o ZIP em local seguro e IMUTÁVEL (Drive + disco local): o INPI guarda só
o hash; numa disputa, é preciso apresentar o arquivo que gera aquele hash.

### Pacote GERADO (16/08/2026) — usar este hash no formulário

O pacote foi montado com `git archive` de HEAD dos 3 repositórios (só código
versionado, sem venv/dados/modelos binários), mais um LEIA-ME com autora, data e
commits (88ff878 / 40e6c60 / 72be247):

- Arquivo: `luciola_codigo_v1.zip` (28,1 MB)
- Cópias imutáveis: `C:\Users\Renato\luciola_inpi\` e `G:\Meu Drive\USP\`
- **SHA-512**: `596151739352B2A07980148121CC80EA8A6A916721279CBF5726A1F914649AE5754B5A03376008D0B125FEBB9CB1C878766C1225A602BCD1B2F9F95334224DE6`

NÃO editar nem re-compactar esse ZIP: qualquer byte diferente muda o hash.

## Passo a passo

1. Cadastro no e-INPI (mesma conta do dossiê de marca).
2. Emitir e pagar GRU código 730.
3. e-Software: https://www.gov.br/inpi/pt-br/servicos/programas-de-computador
   → formulário com: titular/autor (Isabela), data de criação, linguagens
   (Python, JavaScript), campo de aplicação, hash **SHA-512** do pacote e
   algoritmo usado.
4. Assinar a Declaração de Veracidade digitalmente (gov.br assinatura ou
   certificado ICP-Brasil).
5. Guardar o certificado emitido.

## Fontes (consultadas 14/08/2026)

- Serviço oficial: https://www.gov.br/pt-br/servicos/solicitar-o-registro-de-programa-de-computador
- Guia completo: https://www.gov.br/inpi/pt-br/servicos/programas-de-computador/guia-completo-de-programa-de-computador
- FAQ (recomenda SHA-512): https://www.gov.br/inpi/pt-br/acesso-a-informacao/perguntas-frequentes/programas-de-computador
