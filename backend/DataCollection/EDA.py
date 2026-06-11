import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.graphics.tsaplots import plot_pacf , plot_acf

from DataFetch import get_stock_data

df = get_stock_data('reliance.ns','max')

plt.style.use('dark_background')
plt.figure(figsize=(10,6))



# sn.histplot(df['target'],bins=60)
# mean_val = df['target'].mean()
# std_val = df['target'].std()
# plt.axvline(mean_val,color="red",linestyle="--",label=f"Mean: {mean_val:.5f}",linewidth=2,)
# plt.axvline(mean_val + std_val,color="orange",linestyle=":",label=f"+1 STD: {mean_val+std_val:.2f}",)
# plt.axvline(mean_val - std_val,color="orange",linestyle=":",label=f"-1 STD: {mean_val-std_val:.2f}",)
# plt.title("TARGET DISTRIBUTION BOUNDARIES (Volatility-Standardized Z-Score)",color="yellow",fontweight="bold",)
# plt.show()

# ********************************************************************************************************************

# sn.lineplot(df,x=df.index,y='Close',)
# plt.xlabel('year')
# plt.ylabel('price')
# plt.show()

# ********************************************************************************************************************
# plt.plot(df.index, df['realized_vol_20d'], color='crimson',  label='20-Day Rolling Volatility')
# plt.show()
# ********************************************************************************************************************

# plot_acf(df['target'], lags=20 , color='darkblue')
# plot_pacf(df['target'],lags = 20, color = 'darkred')

# ********************************************************************************************************************

sn.heatmap(df.corr(method='spearman'))
plt.show()

# ********************************************************************************************************************

# core_alpha_features = [
#     "target",
#     "RSI",
#     "MACD_diff",
#     "ATR",
#     "vol_stretch",
#     "VWAP_distance",
#     "rolling_beta",
#     "rolling_alpha",
#     "market_breadth",
#     "volatility_ratio_5_20",
# ]
# existing_cols = [c for c in core_alpha_features if c in df.columns]

# corr_mat = df[existing_cols].corr(method= 'spearman')
# sn.heatmap(corr_mat,annot=True )
# plt.title("CORE FEATURE-TO-TARGET CORRELATION MATRIX (Spearman Rank)",color="yellow",fontweight="bold")
# plt.show()

# ********************************************************************************************************************

# plt.plot(df.index,df['target_vol'])
# plt.title("MACRO VOLATILITY STRUCTURAL REGIME DRIFTS (Timeline Clustering)",color="yellow",fontweight="bold")
# plt.show()

# ********************************************************************************************************************
# feature_name: str = "MACD_diff"
# lags = list(range(1, 16))
# correlations = [df[feature_name].autocorr(lag=l) for l in lags]

# plt.bar(lags, correlations, color="gold", edgecolor="orange")
# plt.title(f"5. SIGNAL ALPHA DECAY LAG PROFILE ({feature_name.upper()})",color="yellow",fontweight="bold")
# plt.show()

# ********************************************************************************************************************
# fig, ax = plt.subplots(figsize=(10, 6))


# stats.probplot(df["target"], dist="norm", plot=ax)

# ax.get_lines()[0].set_color("cyan")
# ax.get_lines()[0].set_markersize(4)
# ax.get_lines()[1].set_color("red")
# ax.get_lines()[1].set_linewidth(2)

# plt.title("6. Q-Q TARGET REGIME STABILITY (Fat-Tail Outlier Verification)",color="yellow",fontweight="bold")
# plt.xlabel("Theoretical Normal Quantiles")
# plt.ylabel("Actual Sample Quantiles")
# plt.show()

# ********************************************************************************************************************
# close = df["Close"]
# rolling_peak = close.cummax()
# drawdown = (close - rolling_peak) / rolling_peak

# plt.fill_between(
#     df.index,
#     drawdown * 100,
#     0,
#     color="red",
#     alpha=0.4,
#     label="Historical Price Drawdown %",
# )
# plt.plot(df.index, drawdown * 100, color="crimson", linewidth=1)

# plt.title("7. UNDERLYING EQUITY DRAWDOWN SURFACE RISK MATRIX",color="yellow",fontweight="bold")
# plt.ylabel("Peak-to-Trough Drop Percentage (%)")
# plt.xlabel("Historical Horizon Evolution")
# plt.legend(loc="lower left")
# plt.show()
