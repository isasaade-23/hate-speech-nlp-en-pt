# Licença dos artefatos de modelo (Luciola)

O **código** da Luciola é distribuído sob a GNU Affero General Public License
v3.0 (arquivo `LICENSE`). Este documento cobre outra coisa: os **artefatos de
modelo treinado** distribuídos junto com o código, ou seja, os bundles em
`models/` e `model/` (`*.joblib`, `luciola_linear_v5.json`) e tudo que deriva
deles (vetorizadores, coeficientes, limiares calibrados).

**Esses artefatos não estão sob a AGPL-3.0.**

## Termos

Os artefatos de modelo são liberados para **uso em pesquisa e educação apenas**.
Uso comercial não está autorizado.

Isso não é preferência da autora, é o limite do que ela pode conceder. Os pesos
derivam de conjuntos de dados cujas licenças restringem o uso, e ninguém
sublicencia mais direitos do que recebeu.

| fonte no corpus | licença de origem | efeito sobre os pesos |
|---|---|---|
| HateBR | research-only; uso comercial vedado sem consentimento prévio da Sinch | bloqueia uso comercial |
| ToLD-BR | CC BY-SA 4.0 | exige atribuição; share-alike sobre pesos derivados é juridicamente não testado |
| Fortuna et al. 2019 (pt_fortuna) | "Unknown" na origem | indeterminado |
| tweets_ip | página de origem não recuperável | indeterminado |
| Vidgen et al. 2021 | CC BY 4.0 | exige atribuição |
| HateXplain | MIT | permissiva |
| MultiOFF | Apache 2.0 | permissiva |
| HurtLex (léxico) | CC BY-NC-SA 4.0 | NonCommercial; fica fora do modelo servido |

O detalhamento fica em `methodology/data_provenance.md`, no repositório de
pesquisa, que é a fonte atualizada desta tabela.

## Ao usar os artefatos, você concorda em

1. Usar apenas para pesquisa, ensino ou avaliação técnica.
2. Citar o projeto e as fontes acima que exigem atribuição.
3. Não transformar as predições em decisão automática sobre pessoas sem
   revisão humana (ver a seção de uso responsável do README).
4. Repassar estes mesmos termos a quem receber os artefatos de você.

## Uso comercial

Um modelo comercialmente livre precisa ser retreinado apenas com dados de
licença permissiva (hoje: MultiOFF, Vidgen, HateXplain). Esse modelo ainda não
existe. Para tratar disso, procure a autora.

## Aviso

Este texto descreve as restrições conhecidas em 24/08/2026, de boa-fé, e não é
parecer jurídico. Para decisões que dependam disso, consulte as licenças
originais de cada fonte.

Isabela Venancio da Silva, isasaade23@gmail.com
