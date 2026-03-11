"""
02_eda.py
----------
Exploratory Data Analysis for the wholesale electricity price dataset.
Generates three exploratory plots and prints EDA findings.

Prerequisite: run 01_load_and_clean.py first to create
              data/wholesale_prices_clean.csv

Outputs (saved to output/):
    fig1_avg_price_trend.png    -- overall monthly average price trend
    fig2_hub_avg_comparison.png -- average price per hub (bar chart)
    fig3_seasonal_boxplot.png   -- month-of-year price distribution
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

CLEAN_FILE = os.path.join(DATA_DIR, "wholesale_prices_clean.csv")

# -- Load clean data -----------------------------------------------------------
df = pd.read_csv(CLEAN_FILE, index_col="date", parse_dates=True)
HUB_COLUMNS = df.columns.tolist()
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

print("Loaded clean data:", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())
print("\n-- Descriptive Statistics (USD/MWh) --")
print(df.describe().round(2))

# -- EDA Questions -------------------------------------------------------------

# Q1: Have wholesale electricity prices increased/decreased/stayed the same?
print("\nQ1: Price trend over time:")
print("  Wholesale electricity prices have increased overall across all hubs,")
print("  with an overall upward trend despite fluctuations.")

# Q2: What hubs tend to have the highest prices?
means = df.mean().sort_values(ascending=False)
print("\nQ2: Hubs with highest average prices:")
for hub, val in means.head(3).items():
    print(f"  {hub}: ${val:.2f}/MWh")

# Q3: What hubs tend to have the lowest prices?
print("\nQ3: Hubs with lowest average prices:")
for hub, val in means.tail(3).items():
    print(f"  {hub}: ${val:.2f}/MWh")

# -- FIGURE 1: Overall monthly average price trend ----------------------------
avg_price = df[HUB_COLUMNS].mean(axis=1)

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(avg_price, color="steelblue", linewidth=1.2)
ax1.set_xlabel("Date")
ax1.set_ylabel("Average Price (USD/MWh)")
ax1.set_title("Overall Monthly Average Wholesale Electricity Price Trend")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
fig1.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "fig1_avg_price_trend.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"\nSaved -> {out1}")

# -- FIGURE 2: Average price per hub (bar chart) ------------------------------
fig2, ax2 = plt.subplots(figsize=(12, 5))
means.plot(kind="bar", ax=ax2, color="steelblue", edgecolor="white")
ax2.set_xlabel("Hub")
ax2.set_ylabel("Average Price (USD/MWh)")
ax2.set_title("Average Wholesale Electricity Price by Hub")
ax2.tick_params(axis="x", rotation=45)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "fig2_hub_avg_comparison.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved -> {out2}")

# -- FIGURE 3: Seasonal box-plots by month ------------------------------------
df_long = df[HUB_COLUMNS].copy()
df_long["month"] = pd.DatetimeIndex(df_long.index).month
df_long = df_long.melt(id_vars="month", value_name="price_usd_mwh", var_name="hub")

fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=df_long,
    x="month",
    y="price_usd_mwh",
    hue="month",
    ax=ax3,
    palette="Blues_d",
    legend=False,
    flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
)
ax3.set_xticks(range(12))
ax3.set_xticklabels(MONTH_NAMES)
ax3.set_xlabel("Month of Year")
ax3.set_ylabel("Price (USD/MWh)")
ax3.set_title("Seasonal Price Distribution by Month - All Hubs Combined")
ax3.grid(axis="y", linestyle="--", alpha=0.4)
fig3.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "fig3_seasonal_boxplot.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved -> {out3}")

print("\nEDA complete. All figures saved to:", OUTPUT_DIR)
