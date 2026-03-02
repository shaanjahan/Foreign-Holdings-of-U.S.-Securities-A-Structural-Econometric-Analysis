# Foreign Holdings of U.S. Securities: A Structural Econometric Analysis  
### Lamae A. Maharaj · 2026  

**Data:** U.S. Treasury TIC SHL Historical Survey (`shlhistdat.txt`)  
**Period:** 2002–2024 (annual June observations)  
**Coverage:** 247 reporting economies · 12 security categories · 51,978 observations  

---

Foreign ownership of U.S. securities has evolved far beyond the simple narrative of Japan and China purchasing Treasuries. Over the past two decades, the composition, concentration, and trajectory of foreign demand have shifted in ways that matter for rollover risk, geopolitical exposure, and the structure of U.S. external financing.

This project uses the TIC SHL survey data to answer five focused questions. Each question requires a different econometric framework. The methods were selected because they fit the structure of the problem rather than to display technique. The dataset spans 2002 through 2024 and covers 247 economies.

---

## Running the Analysis

```bash
pip install pandas numpy matplotlib scipy scikit-learn jupyter
jupyter notebook TIC_Analysis_Maharaj.ipynb
```

Place `TIC_data.csv` in the same directory as the notebook before running the analysis.

---

# Findings

---

## 1. Did the 2008 Financial Crisis Permanently Change the Growth Rate of Foreign Treasury Demand?

**Method:** Log linear OLS · Chow structural break test  

Foreign Treasury holdings increased from $0.91 trillion in 2002 to $7.11 trillion in 2024, a cumulative rise of 681 percent. The key question is whether this growth path represents one continuous process or two distinct regimes. A log linear specification allows the slope to be interpreted as an approximate annual growth rate. The Chow test evaluates whether the slope before and after 2008 differs statistically.

| Metric | Pre-2008 | Post-2008 | Full Sample |
|--------|----------|-----------|-------------|
| Annual Growth Rate (CAGR) | **16.29%** | **6.03%** | 9.35% |
| R² | 0.960 | 0.799 | 0.885 |
| p-value | 0.0006 | <0.0001 | 2.5×10⁻¹¹ |
| Chow F-statistic | — | — | **17.59** |
| Chow p-value | — | — | **0.000048** |

The Chow test rejects parameter stability with strong confidence (F = 17.59, p < 0.001). Before 2008, foreign Treasury demand expanded at an annual rate of 16.3 percent. After the crisis, the growth rate fell to 6.0 percent. The level of demand shifted upward permanently, but the pace of growth did not return to its earlier trajectory.

---

## 2. Is the Foreign Creditor Base Becoming More or Less Concentrated?

**Method:** Herfindahl Hirschman Index · linear trend regression  

A concentrated creditor base increases rollover risk. The Herfindahl Hirschman Index measures concentration by summing the squared portfolio shares of each country.

| Metric | Value |
|--------|-------|
| HHI in 2002 | 0.107 |
| HHI in 2024 | 0.057 |
| Total reduction | **47.2%** |
| Annual trend slope | −0.0043 per year |
| R² of trend | 0.700 |
| p-value of trend | **<0.000001** |

Concentration declined by nearly half over the sample period. The downward trend is statistically significant and persistent (p < 0.000001, R² = 0.700). Japan’s share fell from 29 percent to 14 percent. China peaked at 33 percent in 2010 and declined to 11 percent by 2024. The gap has been filled by the United Kingdom, India, Canada, France, and Korea.

---

## 3. China Is Reducing Its Holdings. Is That Supply Being Absorbed?

**Method:** OLS trend regression · Pearson correlation · linear projection  

China’s Treasury portfolio declined from a peak of $1.302 trillion in 2011 to $0.753 trillion in 2024.

**China’s divestment (OLS, 2011–2024):**

| Metric | Value |
|--------|-------|
| Peak holding (June 2011) | $1.302 trillion |
| Holding in June 2024 | $0.753 trillion |
| OLS slope | **−$36.21 billion per year** |
| 95% Confidence Interval | [−$48.1B, −$24.3B] |
| R² | 0.786 |
| p-value | **0.000024** |
| Projected holding in 2030 | $0.657 trillion |

The decline is steady and statistically robust.

**Substitution correlations:**

| Country | Pearson r | p-value | Significance |
|---------|-----------|---------|--------------|
| United Kingdom | −0.702 | 0.00019 | *** |
| France | −0.657 | 0.00066 | *** |
| Canada | −0.589 | 0.00312 | *** |
| Korea, South | −0.549 | 0.00667 | *** |
| India | −0.435 | 0.03799 | ** |
| Japan | −0.104 | 0.63821 | n.s. |
| Taiwan | −0.002 | 0.99244 | n.s. |

`*** p<0.01  ** p<0.05  n.s. = not significant`

The evidence suggests that China’s reduction has been absorbed without a structural demand shortfall.

---

## 4. Has the Foreign Portfolio Shifted from Bonds Toward Equity?

**Method:** Linear trend regression · Welch two sample t-test  

| Metric | Equity Share | Treasury Share |
|--------|-------------|----------------|
| Value in 2002 | 32.17% | 20.93% |
| Value in 2024 | **54.66%** | 23.03% |
| Change (percentage points) | **+22.49 pp** | +2.10 pp |
| Annual trend slope | **+1.051 pp per year** | +0.204 pp per year |
| Trend R² | 0.735 | 0.088 |
| Trend p-value | **<0.000001** | 0.170 (not significant) |
| Early period mean (2002–2010) | 29.85% | — |
| Late period mean (2014–2024) | **43.95%** | — |
| Welch t-statistic | **−6.61** | — |
| Welch p-value | **0.00001** | — |

The equity share rose by 22 percentage points over the sample period and the shift is statistically significant. The Treasury share shows no statistically significant trend.

---

## 5. Do Distinct Country Groups Emerge from the Data?

**Method:** Principal Component Analysis · K Means clustering (k = 4)

**PCA Results:**

| Metric | Value |
|--------|-------|
| Countries in analysis | 73 |
| PC1 variance explained | **93.0%** |
| PC2 variance explained | 3.9% |
| PC3 variance explained | 2.5% |
| PC1–PC3 cumulative | **99.3%** |

**Clusters:**

| Cluster | n | Profile | Notable Members |
|---------|---|---------|-----------------|
| 0 | 15 | Large Western economies | UK, Canada, France, India, Brazil |
| 1 | 1  | Large gradual decliner | **Japan** |
| 2 | 56 | Small to mid sized holders | Australia, Austria, Argentina |
| 3 | 1  | Large strategic reducer | **China** |

Japan and China form distinct singleton clusters, reflecting unique trajectories.

---

## Methods

| Method | Application |
|--------|-------------|
| Log linear OLS | Estimates exponential growth; slope approximates CAGR |
| Chow test | Tests for structural break |
| Herfindahl Hirschman Index | Measures portfolio concentration |
| Pearson correlation | Tests systematic co movement |
| Welch two sample t-test | Compares means without equal variance assumption |
| PCA | Reduces dimensionality |
| K Means clustering | Groups countries by behavioral similarity |

---

## References

- Caballero, R. J., Farhi, E., & Gourinchas, P.-O. (2017). *The safe assets shortage conundrum.* Journal of Economic Perspectives, 31(3), 29–46.  
- Chow, G. C. (1960). *Tests of equality between sets of coefficients in two linear regressions.* Econometrica, 28(3), 591–605.  
- Gourinchas, P.-O., & Rey, H. (2007). *From world banker to world venture capitalist.* University of Chicago Press.  
- Obstfeld, M., & Rogoff, K. (1996). *Foundations of International Macroeconomics.* MIT Press.  
- U.S. Department of the Treasury. TIC System: Annual Survey of Foreign Portfolio Holdings of U.S. Securities.  

---

Lamae A. Maharaj · Independent Research · 2026
