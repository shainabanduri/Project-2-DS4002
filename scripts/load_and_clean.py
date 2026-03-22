"""
load_and_clean.py
---------------------
Dataset establishment: load the raw xlsx, clean missing values,
and save a cleaned CSV.

Outputs:
    data/wholesale_prices_clean.csv  -- cleaned prices (USD/MWh), monthly index
"""

import os
import pandas as pd
import numpy as np

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data")

RAW_FILE   = os.path.join(DATA_DIR, "us-wholesale-electrictiy-prices-monthly.xlsx")
CLEAN_FILE = os.path.join(DATA_DIR, "wholesale_prices_clean.csv")

# -- Column short-names --------------------------------------------------------
HUB_COLUMNS = [
    "ERCOT_North",
    "CAISO_SP15",
    "ISONE_Internal",
    "NYISO_HudsonValley",
    "PJM_Western",
    "MISO_Illinois",
    "SPP_South",
    "SERC_Southern",
    "FRCC_Florida",
    "NW_MidColumbia",
    "SW_PaloVerde",
]

# -- Load ----------------------------------------------------------------------
print("Loading raw data from:", RAW_FILE)

raw = pd.read_excel(
    RAW_FILE,
    sheet_name="Wholesale prices",
    header=1,
    skiprows=[0],
    usecols=range(13),
    engine="openpyxl",
)

raw.columns = ["date_str", "date", *HUB_COLUMNS]
raw = raw[pd.to_datetime(raw["date"], errors="coerce").notna()].copy()

# -- Parse datetime index ------------------------------------------------------
raw["date"] = pd.to_datetime(raw["date"])
raw = raw.set_index("date").drop(columns=["date_str"])
raw.index.name = "date"
raw.index = raw.index.to_period("M").to_timestamp("M")

# -- Replace 'NA' strings with NaN ---------------------------------------------
raw = raw.replace("NA", np.nan)
raw = raw.apply(pd.to_numeric, errors="coerce")

print("\nMissing values per hub (before interpolation):")
print(raw.isna().sum())

# -- Linear interpolation ------------------------------------------------------
df_clean = raw.interpolate(method="time", limit_direction="both")

print("\nMissing values after interpolation:", df_clean.isna().sum().sum())
print("Clean dataset shape:", df_clean.shape)
print("\nDescriptive statistics (USD/MWh):")
print(df_clean.describe().round(2))

# -- Save ----------------------------------------------------------------------
df_clean.to_csv(CLEAN_FILE)
print(f"\nSaved cleaned data -> {CLEAN_FILE}")

# -- Data dictionary -----------------------------------------------------------
print("\nData Dictionary:")
print(f"{'column':<20} {'unit':<10} description")
print("-" * 75)
dict_rows = [
    ("date",               "N/A",     "End-of-month date (monthly frequency)"),
    ("ERCOT_North",        "USD/MWh", "US ERCOT North hub. Missing Jan-Nov 2010; interpolated."),
    ("CAISO_SP15",         "USD/MWh", "US CAISO SP15 zone."),
    ("ISONE_Internal",     "USD/MWh", "US ISO-NE Internal hub."),
    ("NYISO_HudsonValley", "USD/MWh", "US NYISO Hudson Valley zone."),
    ("PJM_Western",        "USD/MWh", "US PJM Western hub."),
    ("MISO_Illinois",      "USD/MWh", "US Midcontinent ISO Illinois hub."),
    ("SPP_South",          "USD/MWh", "US SPP ISO South hub. Missing Jan 2010-Feb 2014; interpolated."),
    ("SERC_Southern",      "USD/MWh", "US SERC index Into Southern."),
    ("FRCC_Florida",       "USD/MWh", "US FRCC index Florida Reliability."),
    ("NW_MidColumbia",     "USD/MWh", "US Northwest index Mid-Columbia."),
    ("SW_PaloVerde",       "USD/MWh", "US Southwest index Palo Verde."),
]
for col, unit, desc in dict_rows:
    print(f"{col:<20} {unit:<10} {desc}")
