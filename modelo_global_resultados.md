# Modelo Global de Preços de Combustíveis — Documentação e Resultados

## 1. Especificação do Modelo

O modelo estimado é uma **regressão log-linear com efeitos fixos categóricos e quebra estrutural**:

```
log(mediana_preco) ~ C(mes) + C(produto) + C(uf) + t + D_pre + t:D_pre
```

A variável dependente é o logaritmo natural da mediana semanal de preço de venda por produto × UF × semana. Transformar a variável em log tem duas consequências diretas: (1) os coeficientes passam a ter interpretação percentual (`(exp(β) - 1) × 100%`), e (2) o modelo torna-se mais robusto a valores extremos e à heterocedasticidade típica de séries de preços.

---

## 2. Variáveis e Motivação Teórica

| Termo | Tipo | Interpretação |
|---|---|---|
| `C(mes)` | Dummies sazonais (ref. = Janeiro) | Efeito de cada mês sobre o preço relativo ao mês de referência |
| `C(produto)` | Dummies de produto (ref. = DIESEL) | Diferencial de preço de cada combustível em relação ao Diesel |
| `C(uf)` | Dummies geográficas (ref. = UF mais frequente) | Diferencial de preço por estado, capturando ICMS, logística e concorrência local |
| `t` | Tendência linear contínua (dias acumulados) | Tendência de longo prazo dos preços após o período de quebra |
| `D_pre` | Dummy de período (1 = antes de 31/07/2022) | Captura o nível médio diferente dos preços antes da desoneração federal |
| `t:D_pre` | Interação tendência × período | Permite que a **inclinação da tendência** seja diferente antes e depois da quebra |

### Quebra estrutural em julho/2022

A variável `D_pre` e a interação `t:D_pre` modelam a quebra estrutural associada à **Lei Complementar 194/2022**, que zerou o ICMS sobre combustíveis em todo o Brasil. O modelo não apenas captura a mudança de nível (via `D_pre`), mas também a mudança na velocidade de variação dos preços (via `t:D_pre`), tornando a especificação mais fiel ao comportamento real da série.

---

## 3. Estratégia de Estimação

O modelo é estimado por **OLS com erros padrão de Newey-West (HAC)**, com janela de 4 lags:

```python
smf.ols(formula, data=train_g).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True})
```

Essa escolha é motivada pela natureza dos dados: séries temporais semanais com autocorrelação serial nos resíduos e possível heterocedasticidade. Os erros HAC são consistentes sob ambos os problemas, mantendo os coeficientes não-viesados e os testes de hipótese válidos.

**Período de treino:** semanas até 31/12/2024  
**Período de teste (holdout):** semanas de 2025

---

## 4. Contribuição Incremental por Bloco de Variáveis (ΔR²)

A tabela abaixo mostra quanto cada bloco de variáveis adiciona ao poder explicativo do modelo:

| Bloco adicionado | R² | R² ajustado | ΔR² |
|---|---|---|---|
| Apenas constante | `[PREENCHER]` | `[PREENCHER]` | — |
| + Tendência (`t`) | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` |
| + Quebra (`D_pre`) | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` |
| + Sazonalidade (`C(mes)`) | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` |
| + Produto (`C(produto)`) | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` |
| + Geografia (`C(uf)`) | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` |

> O ΔR² por bloco responde: *"quanto da variação dos preços é explicada pela sazonalidade? Pelo produto? Pela geografia?"*

---

## 5. Ajuste Global do Modelo

| Métrica | Valor |
|---|---|
| R² (treino) | `[PREENCHER]` |
| R² ajustado (treino) | `[PREENCHER]` |
| N observações (treino) | `[PREENCHER]` |

---

## 6. Diagnósticos dos Resíduos

Os resíduos foram avaliados por quatro testes e quatro gráficos.

### 6.1 Normalidade — Jarque-Bera

| Estatística | p-valor | Assimetria | Curtose |
|---|---|---|---|
| `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` |

> **Interpretação:** p-valor `[< ou >]` 0,05 → hipótese de normalidade `[rejeitada / não rejeitada]`. Em amostras grandes como esta, alguma rejeição é esperada mesmo com resíduos aproximadamente normais. O Q-Q plot com envelope de simulação Monte Carlo (B=300) fornece uma avaliação visual mais informativa.

### 6.2 Heterocedasticidade — Breusch-Pagan e White

| Teste | Estatística | p-valor | Conclusão |
|---|---|---|---|
| Breusch-Pagan | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` |
| White (leve) | `[PREENCHER]` (F) | `[PREENCHER]` | `[PREENCHER]` |

> **Interpretação:** A presença de heterocedasticidade é esperada em dados de painel com múltiplos produtos e estados. Os erros padrão HAC já corrigem esse problema para fins de inferência, de modo que o resultado não invalida os coeficientes estimados.

### 6.3 Gráficos de diagnóstico

- **Histograma dos resíduos:** avalia simetria e caudas da distribuição.
- **Q-Q plot com envelope 95%:** compara os quantis dos resíduos com quantis normais teóricos, com bandas de simulação Monte Carlo. Pontos fora do envelope indicam afastamento da normalidade.
- **Resíduos vs. Ajustados:** avalia estrutura não-linear e heterocedasticidade.
- **Resíduos ao longo do tempo:** detecta autocorrelação, sazonalidade não capturada ou outros padrões temporais remanescentes.

---

## 7. Efeitos Estimados por Grupo

### 7.1 Sazonalidade (efeito % vs. Janeiro)

> Preencher com os coeficientes ou o coefplot gerado. Destacar os meses de pico e vale.

| Mês | Coeficiente (log) | Efeito (%) | IC 95% |
|---|---|---|---|
| Fevereiro | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` |
| ... | ... | ... | ... |

### 7.2 Diferencial por produto (efeito % vs. Diesel)

> Preencher com os coeficientes estimados. Espera-se coeficientes positivos para Gasolina e GNV em relação ao Diesel.

| Produto | Coeficiente (log) | Efeito (%) | IC 95% |
|---|---|---|---|
| `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` |

### 7.3 Diferencial geográfico (efeito % vs. UF de referência)

> Preencher com os estados com maiores desvios positivos e negativos.

---

## 8. Desempenho Preditivo — Holdout 2025

### 8.1 Correção de back-transform (Duan Smearing)

A predição no nível original dos preços não é simplesmente `exp(log_ŷ)`. Como `E[exp(ε)] ≠ 1` quando os resíduos não seguem distribuição normal exata, aplicou-se o **fator de smearing de Duan (1983)**:

```
ŷ_nivel = exp(log_ŷ) × mean(exp(ε_treino))
```

Esse fator é um corretor não-paramétrico de viés, calculado empiricamente sobre os resíduos do treino. Ignorá-lo resultaria em subestimação sistemática dos preços previstos.

| Fator de smearing | `[PREENCHER]` |
|---|---|

### 8.2 Métricas globais (2025)

| Métrica | Valor |
|---|---|
| MAE | `[PREENCHER]` R$/L |
| RMSE | `[PREENCHER]` R$/L |
| MAPE | `[PREENCHER]` % |

### 8.3 Métricas por produto (2025)

| Produto | MAE | RMSE | MAPE (%) | N semanas |
|---|---|---|---|---|
| `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` | `[PREENCHER]` |

---

## 9. Limitações e Considerações

- **Painel não-balanceado:** nem todas as UFs × produtos possuem observações em todas as semanas, o que pode afetar a estabilidade de alguns coeficientes geográficos.
- **Tendência linear:** o modelo assume tendência linear por partes (antes e depois da quebra). Tendências não-lineares (ciclos de commodities, câmbio) não são capturadas diretamente.
- **Ausência de variáveis exógenas:** o modelo não inclui o preço do petróleo (Brent), câmbio ou custos de refino explicitamente — toda essa variação é absorvida pela tendência `t` e pelos efeitos fixos.
- **Heterogeneidade não observada:** efeitos fixos de UF capturam diferenças médias, mas não variações ao longo do tempo dentro de cada estado (ex.: mudanças de alíquota de ICMS estadual).

---

## 10. Referências

- Duan, N. (1983). *Smearing estimate: a nonparametric retransformation method.* Journal of the American Statistical Association, 78(383), 605–610.
- Newey, W. K., & West, K. D. (1987). *A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix.* Econometrica, 55(3), 703–708.
- ANP — Agência Nacional do Petróleo, Gás Natural e Biocombustíveis. Levantamento de Preços de Combustíveis.
