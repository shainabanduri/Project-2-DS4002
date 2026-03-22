"""
modeling.py
--------------
Fit univariate SARIMA and multivariate VAR models on log-transformed monthly
wholesale electricity prices. Compare out-of-sample MAE/RMSE over a 12-month
hold-out window (Jan–Dec 2026), then refit the winning model on all historical
data and forecast Jan–Dec 2027, comparing against EIA STEO projections.

Prerequisites: run load_and_clean.py first.

Outputs (saved to output/):
    sarima_vs_var_mae_by_hub.png       -- MAE by hub: SARIMA vs. VAR
    2027_hub_forecasts_vs_steo.png     -- 2027 hub forecasts vs. EIA STEO
    model_comparison_mae_rmse.csv      -- MAE/RMSE per hub, both models
    2027_mape_vs_steo.csv              -- MAPE vs. EIA STEO 2027 for winning model
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

CLEAN_FILE = os.path.join(DATA_DIR, "wholesale_prices_clean.csv")

# -- Load data -----------------------------------------------------------------
df = pd.read_csv(CLEAN_FILE, index_col="date", parse_dates=True)
df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp("M")

HUB_COLS = df.columns.tolist()

# Historical data: Feb 2010 – Dec 2026
# EIA STEO projections already in the CSV: Jan–Dec 2027
hist      = df[df.index <= "2026-12-31"]
steo_2027 = df[df.index >= "2027-01-01"]   # 12 rows – EIA STEO values

# Train / test split:  train = Feb 2010–Dec 2025,  test = Jan–Dec 2026
train = hist[hist.index <= "2025-12-31"]
test  = hist[hist.index >= "2026-01-01"]

print(f"Train : {train.index.min().date()} to {train.index.max().date()}  ({len(train)} months)")
print(f"Test  : {test.index.min().date()} to {test.index.max().date()}  ({len(test)} months)")
print(f"STEO  : {steo_2027.index.min().date()} to {steo_2027.index.max().date()}  ({len(steo_2027)} months)")

# -- Log-transform prices (reduces right skew, stabilises variance) -----------
# Clip to 0.01 first to guard against the one near-zero/negative CAISO value
log_train = np.log(train.clip(lower=0.01))
log_hist  = np.log(hist.clip(lower=0.01))

# -----------------------------------------------------------------------------
# 1. UNIVARIATE SARIMA  – one model per hub
#    Order (1,1,1)(1,1,1,12): reasonable default for monthly energy time series.
#    d=1 / D=1 handle trending and seasonal non-stationarity.
# -----------------------------------------------------------------------------
SARIMA_ORDER          = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)

sarima_preds = {}
print("\nFitting SARIMA per hub on training data ...")
for hub in HUB_COLS:
    print(f"  {hub} ...", end=" ", flush=True)
    try:
        model = SARIMAX(
            log_train[hub],
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=200)
        # Forecast 12 steps ahead (Jan–Dec 2026); back-transform from log space
        fc = res.forecast(steps=len(test))
        sarima_preds[hub] = np.exp(fc.values)
        print("done")
    except Exception as exc:
        print(f"FAILED ({exc})")
        sarima_preds[hub] = np.full(len(test), np.nan)

sarima_df = pd.DataFrame(sarima_preds, index=test.index)

# -----------------------------------------------------------------------------
# 2. MULTIVARIATE VAR  – joint model for all hubs simultaneously
#    Lag order selected by AIC (up to 4 months; higher values are
#    over-parameterised with 11 hubs).
# -----------------------------------------------------------------------------
print("\nFitting VAR on training data ...")
var_model  = VAR(log_train)
var_lag    = var_model.select_order(maxlags=4).selected_orders["aic"]
var_lag    = max(1, var_lag)           # ensure at least lag-1
print(f"  VAR lag selected by AIC: {var_lag}")
var_res    = var_model.fit(var_lag)

# Forecast 12 steps ahead; seed with last var_lag observations from training
var_fc_log = var_res.forecast(log_train.values[-var_lag:], steps=len(test))
var_df     = pd.DataFrame(np.exp(var_fc_log), index=test.index, columns=HUB_COLS)

# -----------------------------------------------------------------------------
# 3. Evaluation: MAE and RMSE on hold-out set (Jan–Dec 2026)
# -----------------------------------------------------------------------------
def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))

records = []
for hub in HUB_COLS:
    actual = test[hub].values
    records.append({
        "hub":         hub,
        "SARIMA_MAE":  round(mae(actual, sarima_df[hub].values), 4),
        "SARIMA_RMSE": round(rmse(actual, sarima_df[hub].values), 4),
        "VAR_MAE":     round(mae(actual, var_df[hub].values), 4),
        "VAR_RMSE":    round(rmse(actual, var_df[hub].values), 4),
    })

error_df = pd.DataFrame(records).set_index("hub")
print("\n-- Model errors on Jan–Dec 2026 hold-out --")
print(error_df.to_string())

error_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_mae_rmse.csv"))
print(f"\nSaved -> output/model_comparison_mae_rmse.csv")

# Determine winning model (lower average MAE across all hubs)
avg_sarima_mae = error_df["SARIMA_MAE"].mean()
avg_var_mae    = error_df["VAR_MAE"].mean()
winner = "SARIMA" if avg_sarima_mae <= avg_var_mae else "VAR"
print(f"\nAverage MAE – SARIMA: {avg_sarima_mae:.4f}  |  VAR: {avg_var_mae:.4f}")
print(f"Winning model: {winner}")

# -----------------------------------------------------------------------------
# 4. FIGURE 4: grouped bar chart of MAE by hub
# -----------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(13, 5))
x     = np.arange(len(HUB_COLS))
width = 0.38
ax4.bar(x - width / 2, error_df["SARIMA_MAE"], width, label="SARIMA", color="steelblue")
ax4.bar(x + width / 2, error_df["VAR_MAE"],    width, label="VAR",    color="coral")
ax4.set_xticks(x)
ax4.set_xticklabels(HUB_COLS, rotation=45, ha="right")
ax4.set_ylabel("MAE (USD/MWh)")
ax4.set_title("Out-of-Sample MAE by Hub: SARIMA vs. VAR  (Jan–Dec 2026 Hold-out)")
ax4.legend()
ax4.grid(axis="y", linestyle="--", alpha=0.4)
fig4.tight_layout()
out4 = os.path.join(OUTPUT_DIR, "sarima_vs_var_mae_by_hub.png")
fig4.savefig(out4, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"Saved -> {out4}")

# -----------------------------------------------------------------------------
# 5. Refit winning model on all historical data (Feb 2010–Dec 2026)
#    then forecast Jan–Dec 2027
# -----------------------------------------------------------------------------
print(f"\nRefitting {winner} on full historical data for 2027 forecast ...")

if winner == "SARIMA":
    fwd_preds = {}
    for hub in HUB_COLS:
        print(f"  {hub} ...", end=" ", flush=True)
        try:
            model = SARIMAX(
                log_hist[hub],
                order=SARIMA_ORDER,
                seasonal_order=SARIMA_SEASONAL_ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=200)
            fc  = res.forecast(steps=12)
            fwd_preds[hub] = np.exp(fc.values)
            print("done")
        except Exception as exc:
            print(f"FAILED ({exc})")
            fwd_preds[hub] = np.full(12, np.nan)
    fwd_df = pd.DataFrame(fwd_preds, index=steo_2027.index)

else:  # VAR
    var_full  = VAR(log_hist)
    var_lag_f = var_full.select_order(maxlags=4).selected_orders["aic"]
    var_lag_f = max(1, var_lag_f)
    var_res_f = var_full.fit(var_lag_f)
    fc_log    = var_res_f.forecast(log_hist.values[-var_lag_f:], steps=12)
    fwd_df    = pd.DataFrame(np.exp(fc_log), index=steo_2027.index, columns=HUB_COLS)

# -----------------------------------------------------------------------------
# 6. MAPE: model 2027 forecast vs. EIA STEO 2027
# -----------------------------------------------------------------------------
mape_records = []
for hub in HUB_COLS:
    actual = steo_2027[hub].values
    pred   = fwd_df[hub].values
    mape   = np.mean(np.abs((actual - pred) / actual)) * 100
    mape_records.append({"hub": hub, f"{winner}_MAPE_vs_STEO_2027": round(mape, 2)})

mape_df = pd.DataFrame(mape_records).set_index("hub")
print("\n-- 2027 MAPE vs. EIA STEO --")
print(mape_df.to_string())

mape_df.to_csv(os.path.join(OUTPUT_DIR, "2027_mape_vs_steo.csv"))
print(f"\nSaved -> output/2027_mape_vs_steo.csv")

# -----------------------------------------------------------------------------
# 7. FIGURE 5: 2027 forecast vs. EIA STEO (one subplot per hub)
# -----------------------------------------------------------------------------
n_hubs = len(HUB_COLS)
ncols  = 3
nrows  = (n_hubs + ncols - 1) // ncols   # ceiling division

fig5, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5), sharex=True)
axes_flat  = axes.flatten()
months_2027 = steo_2027.index

for i, hub in enumerate(HUB_COLS):
    ax = axes_flat[i]
    ax.plot(months_2027, steo_2027[hub].values, "k-o",  markersize=4, label="EIA STEO 2027")
    ax.plot(months_2027, fwd_df[hub].values,   "b--s", markersize=4, label=f"{winner} Forecast")
    ax.set_title(hub, fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    if i == 0:
        ax.legend(fontsize=7)

# Hide unused subplot panels
for j in range(n_hubs, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig5.suptitle(f"2027 Hub Forecasts ({winner}) vs. EIA STEO Projections", fontsize=13)
fig5.tight_layout()
out5 = os.path.join(OUTPUT_DIR, "2027_hub_forecasts_vs_steo.png")
fig5.savefig(out5, dpi=150, bbox_inches="tight")
plt.close(fig5)
print(f"Saved -> {out5}")

print("\nModeling complete. All outputs saved to:", OUTPUT_DIR)
