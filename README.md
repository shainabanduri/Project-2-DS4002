# U.S. Wholesale Electricity Price Forecasting

**Group:** Model Citizens
**Members:** Shaina Banduri, Neil Parikh, Nishana Dahal
**Course:** DS 4002

---

## Section 1: Software and Platform

| Item | Details |
|------|---------|
| Language | Python 3.10+ |
| Key packages | `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `openpyxl` |
| Platform | Developed on Windows 11; compatible with macOS and Linux |

Install all dependencies:
```
pip install pandas numpy matplotlib seaborn statsmodels openpyxl
```

---

## Section 2: Map of Documentation

```
Project-2-DS4002/
├── README.md
├── LICENSE.md
├── data/
│   ├── us-wholesale-electrictiy-prices-monthly.xlsx   # Raw EIA data
│   ├── wholesale_prices_clean.csv                     # Cleaned, interpolated data
│   └── data_appendix.pdf                              # Data appendix with variables and descriptive statistics
├── scripts/
│   ├── load_and_clean.py   # Load raw XLSX, interpolate missing values, save CSV
│   ├── eda.py              # Exploratory plots (figures 1–3)
│   └── modeling.py         # SARIMA + VAR modeling, evaluation, 2027 forecast
└── output/
    ├── avg_price_trend.png              # Overall price trend over time
    ├── hub_avg_comparison.png           # Average price by hub (bar chart)
    ├── seasonal_price_distribution.png  # Seasonal price distribution by month
    ├── sarima_vs_var_mae_by_hub.png     # SARIMA vs. VAR MAE comparison
    ├── 2027_hub_forecasts_vs_steo.png   # 2027 forecasts vs. EIA STEO
    ├── model_comparison_mae_rmse.csv    # MAE/RMSE per hub for both models
    └── 2027_mape_vs_steo.csv            # 2027 MAPE vs. EIA STEO by hub
```

---

## Section 3: Instructions for Reproducing Results

Run the three scripts in order from the repository root:

**Step 1 — Clean the raw data:**
```
python scripts/load_and_clean.py
```
Loads the raw Excel file, replaces missing values using linear time interpolation, and saves `data/wholesale_prices_clean.csv`.

**Step 2 — Generate exploratory figures:**
```
python scripts/eda.py
```
Produces three plots saved to `output/`.

**Step 3 — Fit models and generate forecasts:**
```
python scripts/modeling.py
```
Fits SARIMA (per hub) and VAR (all hubs jointly) on log-transformed prices, evaluates both on a Jan–Dec 2026 holdout, selects the winning model, forecasts Jan–Dec 2027, and compares against EIA STEO projections. Saves two figures and two CSV tables to `output/`. This step may take a few minutes due to iterative SARIMA fitting.

---

## References

[1] U.S. Energy Information Administration, "Wholesale Electricity and Natural Gas Market Data," EIA. [Online]. Available: https://www.eia.gov/electricity/wholesale/

[2] U.S. Energy Information Administration, "Short-Term Energy Outlook," EIA, 2025. [Online]. Available: https://www.eia.gov/steo/

[3] A. Ganczarek-Gamrot, J. Bunch, and K. Tworek, "Forecasting Electricity Prices Using the SARIMA Model," *Energies*, vol. 18, no. 3, 2025.
