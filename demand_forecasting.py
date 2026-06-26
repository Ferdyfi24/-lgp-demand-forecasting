# ============================================================
# LPG Demand Forecasting Analysis (Dec 2024 – May 2026)
# Pertamina-Authorized LPG 3kg Sub-Agent | Tanah Datar, West Sumatra
# Author: Ferdy Febrian Iskandar
# Tools: Python, Pandas, NumPy, Matplotlib
# ============================================================

# CELL 1: Install & Import
# !pip install pandas numpy matplotlib  # uncomment if needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
print("✅ Libraries loaded.")

# CELL 2: Generate Dataset (same as LPG Distribution project)
MONTHS = pd.date_range(start='2024-12-01', end='2026-05-01', freq='MS')
AREAS = ['Area_A', 'Area_B', 'Area_C']
AREA_BASE = {'Area_A': 440, 'Area_B': 342, 'Area_C': 198}  # total per area/month
SEASONAL = {12:1.15,1:1.18,2:1.10,3:1.00,4:0.95,5:0.92,
            6:0.90,7:0.88,8:0.90,9:0.95,10:1.00,11:1.08}

rows = []
for month in MONTHS:
    sf = SEASONAL[month.month]
    for area in AREAS:
        demand = max(50, int(np.random.normal(AREA_BASE[area]*sf, AREA_BASE[area]*0.10)))
        rows.append({'month': month, 'area': area, 'demand': demand})

df = pd.DataFrame(rows)
total_monthly = df.groupby('month')['demand'].sum().reset_index()
total_monthly.columns = ['month', 'actual']

print(f"✅ Dataset: {len(total_monthly)} months | Total demand: {total_monthly['actual'].sum():,} cylinders")

# CELL 3: Forecasting Methods
# ----------------------------
# Method 1: Simple Moving Average (3-month)
total_monthly['SMA_3'] = total_monthly['actual'].rolling(window=3).mean().round(0)

# Method 2: Weighted Moving Average (weights: 0.5, 0.3, 0.2)
def wma(series, weights=[0.5, 0.3, 0.2]):
    result = [np.nan] * len(series)
    for i in range(len(weights)-1, len(series)):
        result[i] = sum(series.iloc[i-j] * weights[j] for j in range(len(weights)))
    return pd.Series(result, index=series.index)

total_monthly['WMA_3'] = wma(total_monthly['actual']).round(0)

# Method 3: Simple Exponential Smoothing (alpha=0.4)
alpha = 0.4
ses = [total_monthly['actual'].iloc[0]]
for i in range(1, len(total_monthly)):
    ses.append(alpha * total_monthly['actual'].iloc[i] + (1-alpha) * ses[-1])
total_monthly['SES'] = [round(x, 0) for x in ses]

# CELL 4: Forecast Next 3 Months (Jun–Aug 2026)
forecast_months = pd.date_range(start='2026-06-01', periods=3, freq='MS')
last3 = total_monthly['actual'].tail(3).values

sma_forecast  = [round(last3.mean(), 0)] * 3
wma_forecast  = [round(sum(last3[::-1][j] * [0.5,0.3,0.2][j] for j in range(3)), 0)] * 3
ses_last = total_monthly['SES'].iloc[-1]
ses_forecast = []
for i in range(3):
    val = alpha * (last3[-1] if i == 0 else ses_forecast[-1]) + (1-alpha) * ses_last
    ses_forecast.append(round(val, 0))
    ses_last = val

forecast_df = pd.DataFrame({
    'month': forecast_months,
    'SMA_3': sma_forecast,
    'WMA_3': wma_forecast,
    'SES':   ses_forecast
})

print("\n📅 Forecast Jun–Aug 2026:")
print(forecast_df.to_string(index=False))

# CELL 5: Error Metrics (MAE, MAPE)
valid = total_monthly.dropna()
def mae(actual, forecast): return round(np.mean(np.abs(actual - forecast)), 1)
def mape(actual, forecast): return round(np.mean(np.abs((actual - forecast) / actual)) * 100, 2)

print("\n📊 Forecast Accuracy:")
for method in ['SMA_3', 'WMA_3', 'SES']:
    print(f"  {method}: MAE={mae(valid['actual'], valid[method])} | MAPE={mape(valid['actual'], valid[method])}%")

# CELL 6: Plot 1 — Actual vs Forecast Methods
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(total_monthly['month'], total_monthly['actual'], 'o-', color='#1565C0', linewidth=2.2, label='Actual Demand', zorder=5)
ax.plot(total_monthly['month'], total_monthly['SMA_3'], '--', color='#FB8C00', linewidth=1.6, label='SMA-3')
ax.plot(total_monthly['month'], total_monthly['WMA_3'], '--', color='#43A047', linewidth=1.6, label='WMA-3')
ax.plot(total_monthly['month'], total_monthly['SES'],  '--', color='#8E24AA', linewidth=1.6, label=f'SES (α={alpha})')

# Forecast zone
ax.axvline(pd.Timestamp('2026-06-01'), color='gray', linestyle=':', linewidth=1.2, label='Forecast Start')
ax.fill_betweenx([total_monthly['actual'].min()*0.85, total_monthly['actual'].max()*1.1],
                  pd.Timestamp('2026-06-01'), pd.Timestamp('2026-09-01'),
                  alpha=0.08, color='gray')
for method, color in zip(['SMA_3','WMA_3','SES'], ['#FB8C00','#43A047','#8E24AA']):
    ax.plot(forecast_df['month'], forecast_df[method], 's--', color=color, markersize=7, linewidth=1.4)

ax.set_title('LPG Monthly Demand Forecasting — SMA vs WMA vs SES (2024–2026)', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Cylinders Demanded', fontsize=10)
ax.set_xlabel('Month', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('plot_forecast_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_forecast_comparison.png")

# CELL 7: Plot 2 — Area-Level Demand Trend + Forecast
area_monthly = df.groupby(['month','area'])['demand'].sum().reset_index()

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
palette = {'Area_A':'#1565C0','Area_B':'#FB8C00','Area_C':'#43A047'}

for idx, area in enumerate(AREAS):
    ax = axes[idx]
    subset = area_monthly[area_monthly['area']==area].copy()
    subset['SMA_3'] = subset['demand'].rolling(3).mean()
    alpha_a = 0.4
    ses_a = [subset['demand'].iloc[0]]
    for i in range(1, len(subset)):
        ses_a.append(alpha_a*subset['demand'].iloc[i] + (1-alpha_a)*ses_a[-1])
    subset['SES'] = [round(x,0) for x in ses_a]

    ax.plot(subset['month'], subset['demand'], 'o-', color=palette[area], linewidth=2, label='Actual')
    ax.plot(subset['month'], subset['SMA_3'], '--', color='gray', linewidth=1.4, label='SMA-3')
    ax.plot(subset['month'], subset['SES'], ':', color='black', linewidth=1.4, label='SES')
    ax.set_title(f'{area} — Monthly Demand Trend & Smoothing', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cylinders')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.spines[['top','right']].set_visible(False)

plt.suptitle('Area-Level Demand Forecasting Analysis', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_area_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_area_forecast.png")

# CELL 8: Plot 3 — Forecast Error Comparison Bar Chart
methods = ['SMA_3', 'WMA_3', 'SES']
maes  = [mae(valid['actual'], valid[m]) for m in methods]
mapes = [mape(valid['actual'], valid[m]) for m in methods]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
colors = ['#FB8C00','#43A047','#8E24AA']
axes[0].bar(methods, maes, color=colors, edgecolor='white')
for i, v in enumerate(maes):
    axes[0].text(i, v+1, str(v), ha='center', fontweight='bold')
axes[0].set_title('Mean Absolute Error (MAE)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Cylinders')
axes[0].spines[['top','right']].set_visible(False)

axes[1].bar(methods, mapes, color=colors, edgecolor='white')
for i, v in enumerate(mapes):
    axes[1].text(i, v+0.1, f'{v}%', ha='center', fontweight='bold')
axes[1].set_title('Mean Absolute Percentage Error (MAPE)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('MAPE (%)')
axes[1].spines[['top','right']].set_visible(False)

plt.suptitle('Forecast Method Accuracy Comparison', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_forecast_error.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_forecast_error.png")

print("\n✅ Demand Forecasting Analysis complete.")
