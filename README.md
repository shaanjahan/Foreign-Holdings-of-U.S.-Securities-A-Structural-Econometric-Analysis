# Foreign Holdings of U.S. Securities: A Structural Econometric Analysis
### Lamae A. Maharaj · 2026

> **Data:** U.S. Treasury TIC SHL Historical Survey (`shlhistdat.txt`)  
> **Period:** 2002–2024 (annual June snapshots)  
> **Coverage:** 247 reporting economies · 12 security-type categories · 51,978 observations

---

Foreign ownership of U.S. securities has grown from a relatively straightforward story — Japan and China buying Treasuries — into one of the more structurally interesting questions in international finance. Who holds U.S. debt, how that has shifted over two decades, and what it implies for rollover risk, geopolitical exposure, and external financing are questions that deserve more rigorous treatment than they typically receive.

This analysis uses the TIC SHL survey data to work through five specific questions, each requiring a different econometric approach. The data runs from 2002 through 2024 across 247 economies. The methods range from structural break testing to unsupervised clustering — chosen because they fit the questions, not for show.

---

## Running the Analysis

```bash
pip install pandas numpy matplotlib scipy scikit-learn jupyter
jupyter notebook TIC_Analysis_Maharaj.ipynb
```

`TIC_data.csv` should be in the same directory as the notebook.

---

## Findings

---

### 1. Did the 2008 Crisis Permanently Alter the Growth Rate of Foreign Treasury Demand?

**Method:** Log-linear OLS · Chow structural break test

Foreign Treasury holdings grew from $0.91 trillion in 2002 to $7.11 trillion by 2024 — a 683% increase. The question is whether that trajectory is one story or two. Fitting a log-linear model lets us read the slope as an approximate annual growth rate. The Chow test then asks whether the pre- and post-2008 slopes are statistically distinguishable from each other.

| Metric | Pre-2008 | Post-2008 | Full Sample |
|--------|----------|-----------|-------------|
| Annual Growth Rate (CAGR) | **16.29%** | **6.03%** | 9.35% |
| R² | 0.960 | 0.799 | 0.885 |
| p-value | 0.0006 | <0.0001 | 2.5×10⁻¹¹ |
| Chow F-statistic | — | — | **17.59** |
| Chow p-value | — | — | **0.000048** |

The Chow test rejects parameter stability with a high degree of confidence (F = 17.59, p < 0.001). Before 2008, the market for foreign-held Treasuries was growing at 16.3% annually — a pace driven by dollar reserve accumulation in China, Japan, and commodity-exporting economies. That rate fell to 6.0% after the crisis. The level of demand shifted upward permanently, but the growth rate did not recover. This is consistent with the safe-asset shortage literature (Caballero, Farhi & Gourinchas, 2017): the GFC raised the floor of foreign demand while the simultaneous expansion of Treasury supply capped the subsequent growth rate.

---

### 2. Is the Foreign Creditor Base Becoming More or Less Concentrated?

**Method:** Herfindahl-Hirschman Index · linear trend regression

A concentrated creditor base creates rollover risk. If one or two countries hold the majority of foreign-owned debt and either sells aggressively, the market impact can be significant. The HHI — the standard concentration measure in industrial organisation — quantifies this. Each country's share is squared and summed annually; higher scores indicate greater concentration.

| Metric | Value |
|--------|-------|
| HHI in 2002 | 0.107 |
| HHI in 2024 | 0.057 |
| Total reduction | **47.2%** |
| Annual trend slope | −0.0043 per year |
| R² of trend | 0.700 |
| p-value of trend | **<0.000001** |

Concentration has fallen by nearly half over the sample period, with the decline being statistically significant and consistent (p < 0.000001, R² = 0.700). Japan's share fell from 29% to 14%. China's peaked at 33% in 2010 before declining to 11% by 2024. The gap has been filled by the UK, India, Canada, France, and Korea — a more dispersed set of creditors with different policy motivations, time horizons, and domestic constraints. From a debt management standpoint, this diversification is structurally positive, though it also means the Treasury must manage a broader and less predictable set of relationships.

---

### 3. China Is Selling — Is Anyone Absorbing It?

**Method:** OLS trend regression · Pearson correlation · linear projection

China's reduction from a $1.302 trillion peak in 2011 to $0.753 trillion by 2024 is the most-discussed development in foreign Treasury markets. The practical question — whether this creates a demand gap — turns on whether other buyers are systematically stepping in. A negative Pearson correlation between China's portfolio share and another country's share, sustained over the full sample, is direct evidence of substitution.

**China's divestment (OLS, 2011–2024):**

| Metric | Value |
|--------|-------|
| Peak holding (June 2011) | $1.302 trillion |
| Holding in June 2024 | $0.753 trillion |
| OLS slope | **−$36.21 billion/year** |
| 95% Confidence Interval | [−$48.1B, −$24.3B] |
| R² | 0.786 |
| p-value | **0.000024** |
| Projected holding in 2030 | $0.657 trillion |

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

The divestment is real, consistent, and statistically robust — $36 billion per year with a tight confidence interval and an R² of 0.786. At that pace, China's holdings reach roughly $660 billion by 2030. The more important result is on the substitution side. The UK (r = −0.702), France (r = −0.657), Canada (r = −0.589), and Korea (r = −0.549) all show strong, statistically significant negative correlations with China's share. As China's position falls, these economies' shares rise in a systematic way. Japan and Taiwan are uncorrelated with China's trajectory — their holdings are driven by their own exchange rate and monetary policy considerations, not by absorbing Chinese outflows. The overall picture is that the market is handling China's exit without a structural demand shortfall.

---

### 4. Has the Foreign Portfolio Rotated from Bonds to Equity?

**Method:** Linear trend regression · Welch two-sample t-test

The TIC survey captures all U.S. long-term securities, not just Treasuries. Over the past two decades, U.S. equity markets significantly outperformed fixed income, and the global expansion of passive index investing mechanically increased the weight of U.S. stocks in foreign portfolios. Whether this shows up as a statistically meaningful shift in portfolio composition is testable. The Welch t-test compares mean equity share across two eras without assuming equal variance — more appropriate here than a standard t-test given the evident difference in variability between periods.

| Metric | Equity Share | Treasury Share |
|--------|-------------|----------------|
| Value in 2002 | 32.17% | 20.93% |
| Value in 2024 | **54.66%** | 23.03% |
| Change (percentage points) | **+22.49 pp** | +2.10 pp |
| Annual trend slope | **+1.051 pp/year** | +0.204 pp/year |
| Trend R² | 0.735 | 0.088 |
| Trend p-value | **<0.000001** | 0.170 (n.s.) |
| Early period mean (2002–2010) | 29.85% | — |
| Late period mean (2014–2024) | **43.95%** | — |
| Welch t-statistic | **−6.61** | — |
| Welch p-value | **0.00001** | — |

The equity share rose by 22 percentage points over the sample at a rate of +1.05 pp per year — a trend that is highly statistically significant (p < 0.000001, R² = 0.735). The Welch t-test confirms the shift between the early and late periods is not noise (t = −6.61, p < 0.00001). The Treasury share has no statistically significant trend (p = 0.170); it held relatively steady as a fraction of total holdings while Treasuries grew in dollar terms. The implication is that foreign financing of the United States has increasingly taken the form of equity ownership rather than debt — a qualitatively different kind of external liability that carries no rollover risk and aligns foreign investor returns with U.S. economic performance.

---

### 5. Do Distinct Country Groups Emerge from the Data?

**Method:** Principal Component Analysis · K-Means clustering (k = 4, elbow method)

With 247 countries and 23 time periods, the data is high-dimensional enough that patterns are not visible without dimensionality reduction. PCA compresses the country-by-time share matrix into a small number of components that capture the dominant axes of variation. K-Means clustering then operates on those components to group countries by behavioral similarity — no labels assigned in advance.

**PCA:**

| Metric | Value |
|--------|-------|
| Countries in analysis | 73 |
| PC1 variance explained | **93.0%** |
| PC2 variance explained | 3.9% |
| PC3 variance explained | 2.5% |
| PC1–PC3 cumulative | **99.3%** |

**Clusters (k = 4):**

| Cluster | n | Profile | Notable Members |
|---------|---|---------|-----------------|
| 0 | 15 | Large Western economies; growing or stable shares | UK, Canada, France, India, Brazil, Switzerland, Norway |
| 1 | 1  | Mega-holder; long history, gradually declining share | **Japan** |
| 2 | 56 | Small to mid-sized; minimal share of total | Australia, Argentina, Austria, Bahamas, and others |
| 3 | 1  | Strategic reducer; large but consistently falling share | **China** |

PC1 captures 93% of the total variance — essentially a size ranking — while PC2 captures trajectory shape. Japan and China sit on opposite sides of the PC2 axis, reflecting their different dynamics: Japan's long, slow decline from a dominant position versus China's sharp rise and subsequent reversal. K-Means isolates both as singleton clusters, confirming statistically what is evident qualitatively — no other country behaves like either of them. The 15-country Cluster 0 represents the multi-polar structure emerging to fill the space left by China's retreat. Each cluster has different policy sensitivities and different drivers; treating them as one group in a monitoring or engagement framework would obscure more than it reveals.

---

## Methods

| Method | Application |
|--------|-------------|
| Log-linear OLS | Models exponential growth in holdings; slope ≈ CAGR |
| Chow test | Tests whether regression parameters differ across a known break date |
| Herfindahl-Hirschman Index | Measures portfolio concentration as sum of squared shares |
| Pearson correlation | Tests systematic co-movement between two country share series |
| Welch two-sample t-test | Compares period means without equal-variance assumption |
| PCA | Reduces 23-dimensional trajectories to dominant components |
| K-Means clustering | Groups countries by behavioral similarity in PCA space |

---

## Theoretical Grounding

The analysis connects to three bodies of literature. The safe-asset shortage hypothesis (Caballero, Farhi & Gourinchas, 2017) motivates the structural break test: a permanent demand shift after the GFC is the predicted implication of a world where high-quality collateral is structurally scarce. International reserve management theory explains why export-surplus and commodity-exporting economies hold Treasuries as a byproduct of their external sector, and why the HHI decline reflects the broadening of that practice beyond the original Japan/China core. The exorbitant privilege framework (Gourinchas & Rey, 2007) frames the portfolio composition findings: the traditional pattern of the U.S. issuing safe debt while foreigners hold risky equity is partially reversed when foreign equity ownership rises to 55% of total foreign holdings.

---

## References

- Caballero, R. J., Farhi, E., & Gourinchas, P.-O. (2017). *The safe assets shortage conundrum.* Journal of Economic Perspectives, 31(3), 29–46.
- Chow, G. C. (1960). *Tests of equality between sets of coefficients in two linear regressions.* Econometrica, 28(3), 591–605.
- Gourinchas, P.-O., & Rey, H. (2007). *From world banker to world venture capitalist.* In R. Clarida (Ed.), G7 Current Account Imbalances. University of Chicago Press.
- Herfindahl, O. C. (1950). *Concentration in the Steel Industry.* PhD Dissertation, Columbia University.
- Obstfeld, M., & Rogoff, K. (1996). *Foundations of International Macroeconomics.* MIT Press.
- U.S. Department of the Treasury. *TIC System: Annual Survey of Foreign Portfolio Holdings of U.S. Securities.* https://home.treasury.gov/data/treasury-international-capital-tic-system

---

*Lamae A. Maharaj · Independent Research · 2026*
