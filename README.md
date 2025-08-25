MODELO DE PREDIÇÃO DE INADIMPLÊNCIA

PROJETO X-HEALTH

RESUMO 

Este documento apresenta o desenvolvimento completo de um modelo de machine learning para predição de inadimplência em vendas B2B da empresa X-Health. O projeto resultou em um modelo Random Forest com AUC-ROC de 0.849, capaz de identificar 42.4% dos inadimplentes nos 10% de clientes de maior risco, proporcionando significativo valor de negócio através da otimização de estratégias de cobrança e gestão de risco.

SOBRE A EMPRESA X-HEALTH

A X-Health é uma empresa que atua no segmento B2B de dispositivos eletrônicos voltados para a área da saúde. Comercializo equipamentos com amplo espectro de preços e variados níveis de sofisticação tecnológica, atendendo desde pequenas clínicas até grandes hospitais.

MODELO DE NEGÓCIO E PROBLEMA IDENTIFICADO

- MODELO DE VENDAS: Vendas realizadas a crédito, onde o cliente B2B faz o pedido e realiza o pagamento (à vista ou parcelado) em data futura pré-determinada.
- PROBLEMA CRÍTICO: Identifiquei um número crescente e indesejável de inadimplências (defaults), impactando negativamente o fluxo de caixa e a rentabilidade da empresa.
- OBJETIVO DO PROJETO: Desenvolver um algoritmo de machine learning capaz de calcular a probabilidade de inadimplência para cada novo pedido, permitindo tomada de decisão proativa e estratégias de mitigação de risco.

IMPACTO ESPERADO

REDUÇÃO DE INADIMPLÊNCIA:

- Base real: Modelo captura 42.4% dos inadimplentes nos top 10%
- Premissa conservadora: 50% de eficácia nas intervenções preventivas
- Cálculo: 42.4% × 50% = 21.2% dos defaults evitados
- Resultado: 21% de redução na taxa geral (de 16.67% para 13.17%)

ECONOMIA EM COBRANÇA:

- Base real: Top 20% contém ~60% dos inadimplentes
- Lógica: Concentrar recursos onde há 3x mais problemas, e não nos 100% dos clientes
- Resultado: Mesma eficácia com metade dos recursos OU dobrar eficácia com mesmo custo

ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)

CARACTERÍSTICAS DO DATASET

- Fonte dos dados: Combinação de dados internos da X-Health com informações externas (Serasa)
- 117.273 registros × 22 variáveis
- Variável target: 'default' (0 = pagamento em dia, 1 = inadimplência)
- Inadimplência: 16.67% (19.545 casos de default em 117.273 pedidos)

ESTRUTURA DOS DADOS

Cada linha representa um evento de compra de produtos, onde tanto as variáveis internas quanto externas representam uma "fotografia" do cliente no momento da transação.

Tratamento de valores faltantes:

- Valores missing indicados como "missing" no dataset original
- Desenvolvi uma estratégia de imputação considerando o contexto de negócio
- Criei flags indicativas de informações faltantes quando relevante

VARIÁVEIS DISPONÍVEIS

Variáveis internas (comportamento histórico na X-Health):

- default_3months: Quantidade de defaults nos últimos 3 meses
- ioi_3months, ioi_6months, ioi_12months, ioi_36months: Intervalo médio entre pedidos (em dias) nos últimos X meses
- valor_por_vencer: Total em pagamentos a vencer (R$)
- valor_vencido: Total em pagamentos vencidos (R$)
- valor_quitado: Total pago no histórico de compras (R$)
- valor_total_pedido: Valor total do pedido atual (R$)
- forma_pagamento: Método de pagamento escolhido

Variáveis externas (Bureau de crédito Serasa):

- quant_protestos: Quantidade de protestos de títulos
- valor_protestos: Valor total dos protestos (R$)
- quant_acao_judicial: Quantidade de ações judiciais
- acao_judicial_valor: Valor total das ações judiciais (R$)
- participacao_falencia_valor: Valor de participação em falências (R$)
- dividas_vencidas_valor: Valor total de dívidas vencidas (R$)
- dividas_vencidas_qtd: Quantidade de dívidas vencidas
- falencia_concordata_qtd: Quantidade de concordatas

INSIGHTS DA ANÁLISE EXPLORATÓRIA

Durante a análise exploratória, alguns padrões chamaram bastante atenção e acabaram direcionando as decisões que tomei no projeto:

Correlações que fizeram sentido:

- O histórico de pagamentos (valor_quitado) tem uma correlação negativa forte com default (-0.34), confirmando que quem pagou no passado tende a pagar bem no futuro
- Clientes com histórico de maior valor tendem a fazer pedidos maiores (correlação de 0.72), sugerindo uma relação de confiança construída ao longo do tempo
- Os intervalos entre pedidos recentes e históricos se correlacionam (0.58), indicando que existe um padrão de comportamento consistente

Padrões comportamentais interessantes:

- Clientes bons pagadores têm intervalos mais regulares entre pedidos
- Inadimplentes mostram padrões mais imprevisíveis
- A regularidade dos pedidos parece ser um indicador de estabilidade financeira

A questão dos dados faltantes:

Notei que 8,3% dos registros não tinham informação de forma de pagamento. Inicialmente pensei em apenas imputar esses valores, mas percebi que os clientes com essa informação em falta tinham um perfil diferente e uma taxa de inadimplência 50% maior. Isso me levou a criar uma variável específica para capturar esse padrão.

ENGENHARIA DE FEATURES E PRÉ-PROCESSAMENTO

FEATURES CRIADAS

Com base na análise exploratória, criei três novas variáveis que se mostraram muito úteis para o modelo:

1. tem_restricoes - Simplificando as informações do Serasa
    
    Enfrentei 8 variáveis diferentes de crédito (protestos, ações judiciais, dívidas vencidas, etc.) que eram altamente correlacionadas entre si. Em vez de usar todas separadamente, consolidei tudo em uma única pergunta: "esse cliente tem alguma restrição ou não?"
    
    Se o cliente tem qualquer protesto, ação judicial, dívida vencida ou histórico de falência, marco 1, se não, 0.
    
    Clientes com restrições têm 2,8 vezes mais chance de dar calotes e isso cobre 23% da base.
    
2. razao_vencido_total - Contextualizando o valor em atraso
    
    O valor absoluto em atraso sozinho não fazia sentido. Criei uma razão: valor_vencido dividido pelo valor_total_pedido. Isso mostra quanto o atraso representa em relação ao tamanho do negócio atual com o cliente.
    
    Essa variável teve correlação forte (0.41) com inadimplência e valores acima de 0.15 indicam risco elevado.
    
3. forma_pagamento_missing - Quando o que falta também informa
    
    Inicialmente iria apenas preencher os dados faltantes, mas notei que clientes sem informação de forma de pagamento tinham comportamento diferente. A inadimplência era 24% contra 16% da base geral, indicando urgência na venda, processo atípico ou outro padrão relevante.
    

SELEÇÃO FINAL

Para chegar no conjunto final de 9 variáveis, segui um processo estruturado:

- Análise individual: calculei o Information Value de cada variável e testes de significância estatística. Mantive apenas variáveis com IV > 0.02 e p-value < 0.05
- Verificação de redundâncias: analisei correlação entre variáveis e usei VIF < 5 como critério
- Validação prática: testei combinações usando validação cruzada e seleção forward baseada no AUC-ROC, buscando maximizar performance sem overfitting

O conjunto final:

1. valor_quitado (IV: 0.34)
2. ioi_3months (IV: 0.28)
3. ioi_36months (IV: 0.26)
4. valor_total_pedido (IV: 0.22)
5. valor_vencido (IV: 0.18)
6. tem_restricoes (IV: 0.15)
7. razao_vencido_total (IV: 0.12)
8. default_3months (IV: 0.09)
9. forma_pagamento_missing (IV: 0.06)

O que deixei de fora e por quê:

- Variáveis de intervalo 6 e 12 meses eram redundantes
- Variáveis individuais do Serasa foram consolidadas em tem_restricoes
- Algumas variáveis tinham poder preditivo muito baixo (IV < 0.03)

FEATURE IMPORTANCE E INSIGHTS DE NEGÓCIO

O Random Forest mostrou algumas coisas interessantes sobre o que realmente importa para prever inadimplência:

- valor_quitado (24.5%): histórico de pagamentos é o preditor mais forte
- ioi_3months (22.4%): atividade recente é quase tão importante quanto o histórico
- ioi_36months (21.6%): padrão de longo prazo complementa a visão recente
- valor_total_pedido (21.3%): tamanho do pedido atual representa exposição ao risco

Insights:

- Dados internos são mais valiosos que externos: comportamento do cliente na própria X-Health é mais preditivo que informações do Serasa
- Histórico supera situação atual: padrões passados são mais importantes que problemas pontuais
- Regularidade é proteção: clientes com pedidos regulares são mais confiáveis

RECOMENDAÇÕES PRÁTICAS

Como usar o modelo:

- Integrar o score no sistema de pedidos para avaliação em tempo real
- Criar alertas automáticos para pedidos com score > 0.7
- Desenvolver dashboard para acompanhar distribuição de risco

Estratégias de negócio:

- Política de crédito diferenciada: condições mais restritivas para alto risco, benefícios para baixo risco
- Cobrança inteligente: focar recursos nos 20% de maior risco
- Programas de fidelidade para clientes de baixo risco

IMPACTO ESPERADO

A lógica das estimativas:

- Modelo consegue identificar 42,4% dos futuros inadimplentes concentrados em 10% dos clientes de maior risco, concentração 4,2 vezes maior que aleatória
- Cenário conservador: se evitar metade dos defaults desses 10% de maior risco (políticas, garantias, acompanhamento), temos 42,4% × 50% = 21,2% de redução, taxa cai de 16,67% para 13,17%

Economia em cobrança:

- 20% dos clientes concentram 60% dos problemas, faz sentido focar recursos neles
- Resultado: mesma eficácia com menos esforço, ou melhor resultado com o mesmo esforço
