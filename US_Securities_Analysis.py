# U.S. Securities Analysis from the Treasury International Capital Data
# Use of basic machine learning, statistical analysis and basic python code
# By: Lamae A. Maharaj @shaanjahan

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# This is for consistent plot style throughout the notebook
plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'legend.fontsize': 9,
})

COLORS = ['#1a3a5c','#8b1a1a','#2d6a2d','#8b5a1a','#5a1a8b',
          '#1a6b6b','#6b1a4a','#4a8b1a','#8b3a1a','#1a4a8b']

print('Libraries loaded.')
# %%
# Load and inspect the TIC data 
df = pd.read_csv('/Users/lamaemaharaj/Desktop/Economics Projects/Treasury International Capital Project/TIC_data.csv', index_col=0)
df['date'] = pd.to_datetime(df['date'])

print('=== Dataset Overview ===')
print(f'  Rows:              {len(df):,}')
print(f'  Date range:        {df["date"].min().strftime("%b %Y")} → {df["date"].max().strftime("%b %Y")}')
print(f'  Reporting economies: {df["country_name"].nunique()}')
print(f'  Security categories: {df["security_type"].nunique()}')
print(f'\nSecurity types available:')
for s in sorted(df['security_type'].unique()):
    print(f'  {s}')

df.head()
# %%
treas = df[
    (df['security_type'] == 'treasury_debt') &
    (~df['country_name'].str.contains('Total', na=False)) &
    (df['date'] >= '2002-01-01')
].copy()

# Aggregate total foreign Treasury holdings by date
agg = treas.groupby('date')['value_millions_usd'].sum().reset_index()
agg.columns = ['date', 'total_holdings']
agg = agg.sort_values('date').reset_index(drop=True)
agg['t'] = np.arange(len(agg))                         # integer time index
agg['log_holdings'] = np.log(agg['total_holdings'])    # The log transformation is for the growth rate model

print(f'Working panel: {len(treas):,} country-year observations')
print(f'Time series:   {len(agg)} annual observations')
print(f'Total holdings 2002: ${agg["total_holdings"].iloc[0]/1e6:.2f}T')
print(f'Total holdings 2024: ${agg["total_holdings"].iloc[-1]/1e6:.2f}T')
agg.head()
# %%
# Full-sample log-linear trend
slope_full, intercept_full, r_full, p_full, se_full = stats.linregress(
    agg['t'], agg['log_holdings']
)

print('=== Full Sample (2002–2024) ===')
print(f'  β (slope):        {slope_full:.4f}')
print(f'  Implied CAGR:     {(np.exp(slope_full)-1)*100:.2f}% per year')
print(f'  R²:               {r_full**2:.4f}')
print(f'  p-value:          {p_full:.2e}')
# %%
# Sub-period regressions
# 2008 is index 6 in our 23-period series
pre  = agg[agg['date'] < '2008-01-01'].copy()
post = agg[agg['date'] >= '2008-01-01'].copy()

s_pre,  i_pre,  r_pre,  p_pre,  se_pre  = stats.linregress(pre['t'],  pre['log_holdings'])
s_post, i_post, r_post, p_post, se_post = stats.linregress(post['t'], post['log_holdings'])

print('=== Sub-period Comparison ===')
print(f'  Pre-2008  — β={s_pre:.4f}, CAGR={( np.exp(s_pre)-1)*100:.2f}%, R²={r_pre**2:.3f}, p={p_pre:.4f}')
print(f'  Post-2008 — β={s_post:.4f}, CAGR={(np.exp(s_post)-1)*100:.2f}%, R²={r_post**2:.3f}, p={p_post:.4f}')

# Chow test
# Restricted model: single regression on full sample
fitted_full = intercept_full + slope_full * agg['t']
RSS_R = np.sum((agg['log_holdings'] - fitted_full)**2)

# Unrestricted model: These are  separate regressions on each sub-period
RSS_U = (
    np.sum((pre['log_holdings']  - (i_pre  + s_pre  * pre['t']))**2) +
    np.sum((post['log_holdings'] - (i_post + s_post * post['t']))**2)
)

k = 2   # parameters per equation: (intercept + slope)
N = len(agg)
F_chow = ((RSS_R - RSS_U) / k) / (RSS_U / (N - 2 * k))
p_chow = 1 - stats.f.cdf(F_chow, dfn=k, dfd=N - 2*k)

print(f'\n=== Chow Structural Break Test (break at 2008) ===')
print(f'  F-statistic:  {F_chow:.3f}')
print(f'  p-value:      {p_chow:.5f}')
print(f'  Decision:     {"Reject H₀ — structural break confirmed" if p_chow < 0.05 else "Fail to reject H₀"}')
# %%
# Visualizing the structural break 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Levels with sub-period fitted lines
ax = axes[0]
ax.plot(agg['date'], agg['total_holdings']/1e6, 'o-', color=COLORS[0],
        linewidth=2, markersize=5, label='Observed holdings')
ax.axvline(pd.Timestamp('2008-06-01'), color='grey', linestyle=':', linewidth=1.5, label='2008 break')

# Fitted lines in levels space
pre_fit  = np.exp(i_pre  + s_pre  * pre['t'])
post_fit = np.exp(i_post + s_post * post['t'])
ax.plot(pre['date'],  pre_fit/1e6,  '--', color=COLORS[2], linewidth=2,
        label=f'Pre-2008 trend  (CAGR {(np.exp(s_pre)-1)*100:.1f}%)')
ax.plot(post['date'], post_fit/1e6, '--', color=COLORS[1], linewidth=2,
        label=f'Post-2008 trend (CAGR {(np.exp(s_post)-1)*100:.1f}%)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}T'))
ax.set_title('Total Foreign Treasury Holdings')
ax.set_xlabel('Year')
ax.set_ylabel('Trillions USD')
ax.legend()

# Log-linear view (shows regime more clearly)
ax2 = axes[1]
ax2.plot(agg['date'], agg['log_holdings'], 'o-', color=COLORS[0],
         linewidth=2, markersize=5)
ax2.axvline(pd.Timestamp('2008-06-01'), color='grey', linestyle=':', linewidth=1.5)
ax2.plot(pre['date'],  i_pre  + s_pre  * pre['t'],  '--', color=COLORS[2], linewidth=2)
ax2.plot(post['date'], i_post + s_post * post['t'], '--', color=COLORS[1], linewidth=2)
ax2.set_title('Log(Holdings) — Growth Rate Comparison')
ax2.set_xlabel('Year')
ax2.set_ylabel('log(Millions USD)')
ax2.text(pd.Timestamp('2004-01-01'), agg['log_holdings'].min()+0.05,
         f'Pre-2008\nCAGR: {(np.exp(s_pre)-1)*100:.1f}%', fontsize=9, color=COLORS[2])
ax2.text(pd.Timestamp('2016-01-01'), agg['log_holdings'].min()+0.05,
         f'Post-2008\nCAGR: {(np.exp(s_post)-1)*100:.1f}%', fontsize=9, color=COLORS[1])

plt.suptitle('Section 1 — Structural Break in Foreign Treasury Demand (Chow Test)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig1_structural_break.png', bbox_inches='tight')
plt.show()
# %%
# Computing HHI for each year
def compute_hhi(group):
    """Compute HHI = sum of squared market shares for a given year."""
    total = group['value_millions_usd'].sum()
    if total == 0:
        return np.nan
    shares = group['value_millions_usd'] / total
    return (shares ** 2).sum()

hhi_ts = treas.groupby('date').apply(compute_hhi).reset_index()
hhi_ts.columns = ['date', 'hhi']
hhi_ts = hhi_ts.sort_values('date').reset_index(drop=True)
hhi_ts['t'] = np.arange(len(hhi_ts))

# Linear trend regression on HHI
slope_hhi, intercept_hhi, r_hhi, p_hhi, se_hhi = stats.linregress(
    hhi_ts['t'], hhi_ts['hhi']
)

print('=== HHI Summary ===')
print(f'  2002 HHI:          {hhi_ts["hhi"].iloc[0]:.4f}   ← more concentrated')
print(f'  2024 HHI:          {hhi_ts["hhi"].iloc[-1]:.4f}   ← less concentrated')
print(f'  Total reduction:   {(1-hhi_ts["hhi"].iloc[-1]/hhi_ts["hhi"].iloc[0])*100:.1f}%')
print(f'\n=== Linear Trend ===')
print(f'  Slope:             {slope_hhi:.5f} per year')
print(f'  R²:                {r_hhi**2:.3f}')
print(f'  p-value:           {p_hhi:.4f}')
print(f'  Decision:          {"Statistically significant downward trend" if p_hhi < 0.05 else "Trend not significant"}')
# %%
# Top 5 holders share over time 
# This shows who is driving the concentration changes
total_by_date = treas.groupby('date')['value_millions_usd'].sum()
treas['share_pct'] = (treas['value_millions_usd']
                      / treas['date'].map(total_by_date) * 100)

top5 = ['Japan', 'China (20)', 'United Kingdom', 'Brazil', 'Taiwan']
share_pivot = (treas[treas['country_name'].isin(top5)]
               .pivot_table(index='date', columns='country_name',
                            values='share_pct', aggfunc='sum')
               .fillna(0))

# Plotting HHI + share decomposition 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(hhi_ts['date'], hhi_ts['hhi'], 'o-', color=COLORS[0],
        linewidth=2, markersize=5, label='HHI (observed)')
fitted_hhi = intercept_hhi + slope_hhi * hhi_ts['t']
ax.plot(hhi_ts['date'], fitted_hhi, '--', color=COLORS[1],
        linewidth=2, label=f'Linear trend (β={slope_hhi:.4f}, p<0.001)')
ax.set_title('Herfindahl-Hirschman Index (HHI) of Foreign\nTreasury Ownership Concentration')
ax.set_xlabel('Year')
ax.set_ylabel('HHI  (0 = dispersed → 1 = monopoly)')
ax.legend()
ax.set_ylim(0, None)

ax2 = axes[1]
for i, country in enumerate(top5):
    if country in share_pivot.columns:
        ax2.plot(share_pivot.index, share_pivot[country], 'o-',
                 color=COLORS[i], linewidth=2, markersize=4, label=country)
ax2.set_title('Share of Total Foreign Treasury Holdings\n(Top 5 Economies)')
ax2.set_xlabel('Year')
ax2.set_ylabel('Share of total (%)')
ax2.legend()

plt.suptitle('Section 2 — Concentration Risk: HHI and Share Dynamics',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig2_concentration.png', bbox_inches='tight')
plt.show()
print(share_pivot.round(1).to_string())
# %%
# China divestment OLS (post-2011 peak) 
china = treas[treas['country_name'] == 'China (20)'].sort_values('date').copy()
china_post_peak = china[china['date'] >= '2011-01-01'].copy()
china_post_peak['t'] = np.arange(len(china_post_peak))

s_cn, i_cn, r_cn, p_cn, se_cn = stats.linregress(
    china_post_peak['t'], china_post_peak['value_millions_usd']
)

n_cn = len(china_post_peak)
t_crit = stats.t.ppf(0.975, n_cn - 2)
ci_slope = t_crit * se_cn

print('=== China Post-Peak Divestment (2011–2024) ===')
print(f'  Peak holding (2011):      ${china_post_peak["value_millions_usd"].iloc[0]/1e6:.3f}T')
print(f'  Latest holding (2024):    ${china_post_peak["value_millions_usd"].iloc[-1]/1e6:.3f}T')
print(f'  OLS slope:                ${s_cn/1e3:.1f}B per year')
print(f'  95% CI:                   [${(s_cn-ci_slope)/1e3:.1f}B, ${(s_cn+ci_slope)/1e3:.1f}B]')
print(f'  R²:                       {r_cn**2:.3f}')
print(f'  p-value:                  {p_cn:.5f}')
# Projecting to 2030 (6 more periods)
last_t = china_post_peak['t'].iloc[-1]
proj_2030 = i_cn + s_cn * (last_t + 6)
print(f'  Projected 2030 holding:   ${proj_2030/1e6:.3f}T  (linear extrapolation)')
# %%
# The Substitution Effect: correlation of China's share vs candidate substitutes
china_share = treas[treas['country_name'] == 'China (20)'].set_index('date')['share_pct']

candidates = [
    'United Kingdom', 'India', 'Canada', 'France',
    'Japan', 'Taiwan', 'Norway', 'Korea, South'
]

print('=== Pearson Correlation: China Share vs Candidate Substitutes ===')
print(f'{"Country":<22} {"r":>8} {"p-value":>10} {"Significant?":>14} {"Direction":>12}')
print('-' * 70)

results = []
for country in candidates:
    other_share = treas[treas['country_name'] == country].set_index('date')['share_pct']
    aligned = pd.concat([china_share, other_share], axis=1).dropna()
    if len(aligned) < 5:
        continue
    aligned.columns = ['china', 'other']
    r, p = stats.pearsonr(aligned['china'], aligned['other'])
    sig = '*** (1%)' if p < 0.01 else '** (5%)' if p < 0.05 else '* (10%)' if p < 0.1 else 'n.s.'
    direction = 'Substitution' if r < -0.3 else ('Co-movement' if r > 0.3 else 'Neutral')
    results.append({'country': country, 'r': r, 'p': p})
    print(f'{country:<22} {r:>8.3f} {p:>10.4f} {sig:>14} {direction:>12}')
# %%
# Plot for China trend + projection, and share substitution 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# China observed + fitted trend + projection
ax = axes[0]
ax.plot(china['date'], china['value_millions_usd']/1e6, 'o-',
        color=COLORS[1], linewidth=2, markersize=5, label='China (observed)')

# Fitted trend on post-peak data
fit_dates = china_post_peak['date']
fit_vals  = i_cn + s_cn * china_post_peak['t']
ax.plot(fit_dates, fit_vals/1e6, '--', color=COLORS[3], linewidth=2,
        label=f'OLS trend (slope: ${s_cn/1e3:.0f}B/yr)')

# Projection to 2030
proj_t = np.arange(last_t + 1, last_t + 7)
proj_dates = pd.date_range('2025-06-01', periods=6, freq='YS-JUN')
proj_vals  = i_cn + s_cn * proj_t
proj_ci    = t_crit * se_cn * np.sqrt(1 + 1/n_cn + (proj_t - china_post_peak['t'].mean())**2
                                       / np.sum((china_post_peak['t'] - china_post_peak['t'].mean())**2))
ax.plot(proj_dates, proj_vals/1e6, ':', color=COLORS[3], linewidth=2, label='Projection to 2030')
ax.fill_between(proj_dates,
                (proj_vals - proj_ci)/1e6,
                (proj_vals + proj_ci)/1e6,
                alpha=0.2, color=COLORS[3], label='95% prediction interval')
ax.axvline(pd.Timestamp('2011-06-01'), color='grey', linestyle=':', linewidth=1.2, label='Peak (2011)')
ax.axvline(pd.Timestamp('2022-06-01'), color='red', linestyle=':', linewidth=1.2, alpha=0.5,
           label='Russia sanctions (2022)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.1f}T'))
ax.set_title("China's Treasury Holdings: Trend and Projection")
ax.set_xlabel('Year')
ax.legend(fontsize=8)

# Share substitution — China vs UK, India, Canada
ax2 = axes[1]
sub_countries = ['China (20)', 'United Kingdom', 'India', 'Canada']
sub_colors = [COLORS[1], COLORS[0], COLORS[2], COLORS[3]]
for country, color in zip(sub_countries, sub_colors):
    s = treas[treas['country_name'] == country].set_index('date')['share_pct']
    ax2.plot(s.index, s.values, 'o-', color=color, linewidth=2,
             markersize=4, label=country)
ax2.set_title('Portfolio Share: China Decline vs.\nCandidate Substitutes')
ax2.set_xlabel('Year')
ax2.set_ylabel('Share of total foreign Treasury holdings (%)')
ax2.legend()

plt.suptitle('Section 3 — China Divestment Trend and Substitution Effects',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig3_china_divestment.png', bbox_inches='tight')
plt.show()
# %%
# Portfolio Composition Table
# Using the 'Total' row which aggregates across all reporting economies
total_row = df[
    (df['country_name'] == 'Total') &
    (df['date'] >= '2002-01-01')
].copy()

comp = total_row.pivot_table(
    index='date', columns='security_type', values='value_millions_usd'
).reset_index().sort_values('date')

# Computing shares of total holdings
for col in ['equity', 'treasury_debt', 'agency_debt', 'corporate_debt', 'total_st_debt']:
    if col in comp.columns:
        comp[f'{col}_share'] = comp[col] / comp['total_securities'] * 100

comp['t'] = np.arange(len(comp))

print('=== Portfolio Composition (% of Total Foreign Holdings) ===')
display_cols = ['date','equity_share','treasury_debt_share','agency_debt_share',
                'corporate_debt_share','total_st_debt_share']
display_cols = [c for c in display_cols if c in comp.columns]
print(comp[display_cols].set_index('date').round(1).to_string())
# %%
# Trend regression on equity shares 
s_eq, i_eq, r_eq, p_eq, _ = stats.linregress(comp['t'], comp['equity_share'])
s_tr, i_tr, r_tr, p_tr, _ = stats.linregress(comp['t'], comp['treasury_debt_share'])

print('=== Trend Regressions on Portfolio Shares ===')
print(f'  Equity share:   β={s_eq:.3f} pp/yr, R²={r_eq**2:.3f}, p={p_eq:.4f}')
print(f'  Treasury share: β={s_tr:.3f} pp/yr, R²={r_tr**2:.3f}, p={p_tr:.4f}')

# Two-sample t-test: Early vs Late period 
early = comp[comp['date'] <= '2010-06-01']['equity_share']
late  = comp[comp['date'] >= '2014-06-01']['equity_share']

t_stat, p_ttest = stats.ttest_ind(early, late, equal_var=False)  # Welch's t-test

print(f'\n=== Welch Two-Sample t-test: Equity Share (Early 2002–2010 vs Late 2014–2024) ===')
print(f'  Early period mean:  {early.mean():.1f}%')
print(f'  Late period mean:   {late.mean():.1f}%')
print(f'  Difference:         {late.mean()-early.mean():.1f} percentage points')
print(f'  t-statistic:        {t_stat:.3f}')
print(f'  p-value:            {p_ttest:.4f}')
print(f'  Decision:           {"Reject H₀ — means are significantly different" if p_ttest < 0.05 else "Fail to reject H₀"}')
# %%
# Stacked area chart of portfolio composition 
stack_cols = ['equity', 'treasury_debt', 'corporate_debt', 'agency_debt', 'total_st_debt']
stack_cols = [c for c in stack_cols if c in comp.columns]
stack_labels = {
    'equity': 'Equity',
    'treasury_debt': 'Treasury Debt',
    'corporate_debt': 'Corporate Debt',
    'agency_debt': 'Agency Debt',
    'total_st_debt': 'Short-Term'
}
stack_colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[3], COLORS[4]]

stack_data = comp[stack_cols].div(comp['total_securities'], axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stacked Area Visualization
ax = axes[0]
ax.stackplot(comp['date'],
             [stack_data[c] for c in stack_cols],
             labels=[stack_labels[c] for c in stack_cols],
             colors=stack_colors, alpha=0.85)
ax.set_title('Portfolio Composition of All Foreign\nHoldings of U.S. Securities')
ax.set_xlabel('Year')
ax.set_ylabel('Share of total holdings (%)')
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=8)

# Equity vs Treasury share with trend lines
ax2 = axes[1]
ax2.plot(comp['date'], comp['equity_share'], 'o-', color=COLORS[0],
         linewidth=2, markersize=5, label='Equity share')
ax2.plot(comp['date'], comp['treasury_debt_share'], 'o-', color=COLORS[1],
         linewidth=2, markersize=5, label='Treasury share')
ax2.plot(comp['date'], i_eq + s_eq * comp['t'], '--', color=COLORS[0],
         linewidth=1.5, alpha=0.7, label=f'Equity trend (β={s_eq:.2f} pp/yr, p={p_eq:.3f})')
ax2.plot(comp['date'], i_tr + s_tr * comp['t'], '--', color=COLORS[1],
         linewidth=1.5, alpha=0.7, label=f'Treasury trend (β={s_tr:.2f} pp/yr, p={p_tr:.3f})')
ax2.set_title('Equity vs Treasury Share:\nTrend Lines and Regime Comparison')
ax2.set_xlabel('Year')
ax2.set_ylabel('Share of total holdings (%)')
ax2.legend(fontsize=8)

plt.suptitle('Section 4 — Portfolio Rotation: Equity vs. Fixed Income in Foreign Holdings',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig4_portfolio_composition.png', bbox_inches='tight')
plt.show()
# %%
# Build the country × time share matrix 
total_by_date = treas.groupby('date')['value_millions_usd'].sum()
treas['share'] = treas['value_millions_usd'] / treas['date'].map(total_by_date)

share_pivot = treas.pivot_table(
    index='country_name', columns='date', values='share', aggfunc='sum'
).fillna(0)

# Keeping only the countries with meaningful presence (at least 0.1% at some point)
share_pivot = share_pivot[share_pivot.max(axis=1) >= 0.001]

print(f'Countries in analysis: {len(share_pivot)}')
print(f'Time periods:          {share_pivot.shape[1]}')

# Standardize for PCA/clustering
scaler = StandardScaler()
X = scaler.fit_transform(share_pivot)

# PCA 
pca = PCA(n_components=5, random_state=42)
pca_scores = pca.fit_transform(X)

evr = pca.explained_variance_ratio_
print(f'\n=== PCA Explained Variance ===')
for i, v in enumerate(evr):
    print(f'  PC{i+1}: {v*100:.1f}%  (cumulative: {evr[:i+1].sum()*100:.1f}%)')
# %%
# K-Means clustering 

inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km.fit(pca_scores[:, :3])   # cluster on first 3 PCs
    inertias.append(km.inertia_)

# Fitted with k=4 (natural groupings: mega-holders, large holders, medium, small)
km4 = KMeans(n_clusters=4, random_state=42, n_init=20)
labels = km4.fit_predict(pca_scores[:, :3])

cluster_df = pd.DataFrame({
    'country': share_pivot.index,
    'cluster': labels,
    'PC1': pca_scores[:, 0],
    'PC2': pca_scores[:, 1],
})

# Assigning meaningful labels based on who is in each cluster
for c in sorted(cluster_df['cluster'].unique()):
    members = cluster_df[cluster_df['cluster'] == c]['country'].tolist()
    print(f'Cluster {c} ({len(members)} countries):')
    print('  ', members[:15])
    print()
# %%
# Plot: elbow curve + PCA scatter with clusters 
cluster_names = {}  
# Heuristic labels — update after inspecting your cluster output
for c in sorted(cluster_df['cluster'].unique()):
    members = cluster_df[cluster_df['cluster'] == c]['country'].tolist()
    avg_share = share_pivot.loc[members].mean().mean()
    cluster_names[c] = f'Cluster {c} (n={len(members)})'

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: elbow curve
ax = axes[0]
ax.plot(list(K_range), inertias, 'o-', color=COLORS[0], linewidth=2, markersize=6)
ax.axvline(4, color='grey', linestyle=':', linewidth=1.5, label='Chosen k=4')
ax.set_title('K-Means Elbow Curve\n(Inertia vs. Number of Clusters)')
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Inertia (within-cluster SS)')
ax.legend()

# Panel B: PCA scatter colored by cluster
ax2 = axes[1]
for c in sorted(cluster_df['cluster'].unique()):
    mask = cluster_df['cluster'] == c
    ax2.scatter(cluster_df[mask]['PC1'], cluster_df[mask]['PC2'],
                color=COLORS[c], label=cluster_names[c], s=50, alpha=0.8)
    # Label notable countries
    notable = ['Japan', 'China (20)', 'United Kingdom', 'Brazil', 'Germany', 'India']
    for _, row in cluster_df[mask & cluster_df['country'].isin(notable)].iterrows():
        ax2.annotate(row['country'].replace(' (20)', '').replace(' (5)', ''),
                     (row['PC1'], row['PC2']), fontsize=7,
                     xytext=(4, 4), textcoords='offset points')

ax2.set_title(f'PCA Biplot — Country Clusters\n(PC1: {evr[0]*100:.0f}% var, PC2: {evr[1]*100:.0f}% var)')
ax2.set_xlabel(f'PC1 ({evr[0]*100:.1f}% explained variance)')
ax2.set_ylabel(f'PC2 ({evr[1]*100:.1f}% explained variance)')
ax2.legend(fontsize=8)

plt.suptitle('Section 5 — PCA + K-Means: Latent Groups in Country Holding Trajectories',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig5_pca_clusters.png', bbox_inches='tight')
plt.show()
# %%
# ── Average time-series trajectory per cluster ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

for c in sorted(cluster_df['cluster'].unique()):
    members = cluster_df[cluster_df['cluster'] == c]['country'].tolist()
    avg_trajectory = share_pivot.loc[
        [m for m in members if m in share_pivot.index]
    ].mean(axis=0)
    ax.plot(avg_trajectory.index, avg_trajectory.values * 100,
            'o-', color=COLORS[c], linewidth=2, markersize=4,
            label=cluster_names[c])

ax.set_title('Average Portfolio Share Trajectory by Cluster\n(Cluster-Mean Share of Total Foreign Treasury Holdings)')
ax.set_xlabel('Year')
ax.set_ylabel('Mean share of total foreign Treasury holdings (%)')
ax.legend()
plt.tight_layout()
plt.savefig('fig5b_cluster_trajectories.png', bbox_inches='tight')
plt.show()
# %%
