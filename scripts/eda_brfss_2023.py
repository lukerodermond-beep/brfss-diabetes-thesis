"""
BRFSS 2023 EDA Script (Master Thesis)
Target: DIABETE4

Predictors (from proposal, mapped to BRFSS 2023 variables confirmed via codebook + dataset):
- age                 -> _AGE80
- sex                 -> SEXVAR
- education           -> EDUCA
- income              -> INCOME3
- smoking status      -> _SMOKER3
- alcohol consumption -> ALCDAY4
- physical activity   -> _TOTINDA
- BMI                 -> _BMI5
- hypertension status -> BPHIGH6
- cholesterol status  -> TOLDHI3

NOTE: Sleep duration is excluded because no sleep-duration variable is present in this LLCP2023.XPT file
(verified by searching df.columns for 'SLEP' and 'SLEEP').

How to use:
1) Put this script in the SAME folder as LLCP2023.XPT
2) Open that folder in VS Code
3) Run: python eda_brfss_2023.py

Outputs:
- eda_outputs/ (tables + text summary + plots)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# 0) CONFIG
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "LLCP2023.XPT"

OUT_DIR = SCRIPT_DIR / "eda_outputs"
OUT_DIR.mkdir(exist_ok=True)

TARGET = "DIABETE4"

PREDICTOR_MAP = {
    "age": "_AGE80",
    "sex": "SEXVAR",
    "education": "EDUCA",
    "income": "INCOME3",
    "smoking_status": "_SMOKER3",
    "alcohol_days_past30": "ALCDAY4",
    "physical_activity": "_TOTINDA",
    "bmi": "_BMI5",
    "hypertension_status": "BPHIGH6",
    "cholesterol_status": "TOLDHI3",
}

LEAKAGE_KEYWORDS = ["DIAB", "INSUL", "A1C", "GLUC", "METFORM", "MED", "MEDS"]


# -----------------------
# 1) HELPERS
# -----------------------
def safe_value_counts(series: pd.Series, top_n: int = 15) -> pd.DataFrame:
    vc = series.value_counts(dropna=False)
    out = vc.head(top_n).to_frame(name="count")
    out["percent"] = (out["count"] / len(series) * 100).round(3)
    return out

def missing_summary(df_in: pd.DataFrame) -> pd.DataFrame:
    miss = df_in.isna().mean().sort_values(ascending=False)
    return pd.DataFrame({"missing_rate": miss, "missing_percent": (miss * 100).round(3)})

def to_csv(df_out: pd.DataFrame, filename: str) -> None:
    df_out.to_csv(OUT_DIR / filename, index=True)

def write_text(filename: str, text: str) -> None:
    (OUT_DIR / filename).write_text(text, encoding="utf-8")


# -----------------------
# 2) LOAD DATA
# -----------------------
print("Loading data...")
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}. Make sure LLCP2023.XPT is in the same folder as this script."
    )

df = pd.read_sas(DATA_FILE, format="xport")
print("Loaded:", df.shape)

# Standardize column names
df.columns = [str(c).upper() for c in df.columns]
TARGET = TARGET.upper()

# Uppercase predictor names for matching
predictors_all = {k: v.upper() for k, v in PREDICTOR_MAP.items()}
predictor_cols = list(predictors_all.values())

predictors_found = [c for c in predictor_cols if c in df.columns]
predictors_missing = [c for c in predictor_cols if c not in df.columns]


# -----------------------
# 3) STEP 1 — DATASET OVERVIEW
# -----------------------
overview_lines = [
    f"n_rows: {df.shape[0]}",
    f"n_cols: {df.shape[1]}",
    f"n_predictors_expected: {len(predictor_cols)}",
    f"n_predictors_found: {len(predictors_found)}",
    f"n_predictors_missing: {len(predictors_missing)}",
    f"dtypes_counts: {df.dtypes.value_counts().to_dict()}",
]
if predictors_missing:
    overview_lines.append(f"WARNING missing predictor columns: {predictors_missing}")

write_text("01_overview.txt", "\n".join(overview_lines))

dtypes_tbl = df.dtypes.astype(str).to_frame("dtype")
to_csv(dtypes_tbl, "01_variable_dtypes.csv")

mapping_tbl = pd.DataFrame(
    [{"concept": k, "brfss_variable": v, "exists_in_data": (v in df.columns)}
     for k, v in predictors_all.items()]
)
to_csv(mapping_tbl, "01_predictor_mapping.csv")


# -----------------------
# 4) STEP 2 — TARGET ANALYSIS (DIABETE4)
# -----------------------
if TARGET not in df.columns:
    raise ValueError(f"Target column {TARGET} not found. Check your dataset columns.")

# Raw target distribution
target_counts = safe_value_counts(df[TARGET], top_n=30)
to_csv(target_counts, "02_target_value_counts.csv")

# Binary target: 1 = diabetes, 3 = no diabetes; others excluded
target_binary = df[TARGET].replace({1: 1, 3: 0})
target_binary = target_binary.where(target_binary.isin([0, 1]), np.nan)

target_binary_counts = safe_value_counts(target_binary, top_n=10)
to_csv(target_binary_counts, "02_target_binary_value_counts.csv")

yes = int((target_binary == 1).sum())
no = int((target_binary == 0).sum())

imbalance_lines = [
    f"Binary DIABETE4 counts (excluding other categories): Diabetes(1)={yes}, No diabetes(0)={no}"
]
if yes > 0 and no > 0:
    imbalance_lines.append(f"Imbalance ratio (No/Yes): {no/yes:.2f}")
write_text("02_target_imbalance.txt", "\n".join(imbalance_lines))


# -----------------------
# 5) STEP 3 — DATA QUALITY CHECKS
# -----------------------
miss_all = missing_summary(df)
to_csv(miss_all.head(50), "03_missing_top50.csv")

cols_to_check = [TARGET] + predictors_found
miss_focus = missing_summary(df[cols_to_check])
to_csv(miss_focus, "03_missing_focus_predictors_target.csv")

for col in predictors_found:
    vc = safe_value_counts(df[col], top_n=20)
    to_csv(vc, f"03_value_counts_{col}.csv")


# -----------------------
# 6) STEP 4 — OUTLIERS + BASIC NUMERIC STATS
# -----------------------
numeric_interest = [c for c in ["_AGE80", "_BMI5", "ALCDAY4"] if c in df.columns]
if numeric_interest:
    desc = df[numeric_interest].describe(percentiles=[.01, .05, .5, .95, .99]).T
    to_csv(desc, "04_numeric_descriptives.csv")


# -----------------------
# 7) STEP 5 — PATTERNS / PLAUSIBILITY
# -----------------------
# IMPORTANT: focus contains TARGET_BIN (so plots won't error)
focus = df[[TARGET] + predictors_found].copy()
focus["TARGET_BIN"] = target_binary

import matplotlib.pyplot as plt

# Only keep rows where target is defined
plot_df = focus.dropna(subset=["TARGET_BIN"]).copy()

# -------------------------------------------------
# Fix BMI scale (BRFSS stores BMI * 100)
# -------------------------------------------------
if "_BMI5" in plot_df.columns:
    plot_df["_BMI5_real"] = plot_df["_BMI5"] / 100


# -------------------------------------------------
# Helper function for clean boxplots
# -------------------------------------------------
def make_boxplot(df, value_col, title, ylabel, filename):

    no_diab = df.loc[df["TARGET_BIN"] == 0, value_col].dropna()
    diab    = df.loc[df["TARGET_BIN"] == 1, value_col].dropna()

    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [no_diab, diab],
        labels=["No diabetes", "Diabetes"],
        showfliers=False   # <-- IMPORTANT: remove extreme dots
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.sa


# Numeric descriptives by target
group_tables = []
for num_col in numeric_interest:
    if num_col in focus.columns:
        grp = focus.groupby("TARGET_BIN")[num_col].describe()
        grp.index = grp.index.map({1.0: "Diabetes(1)", 0.0: "No diabetes(0)"})
        grp["variable"] = num_col
        group_tables.append(grp.reset_index().rename(columns={"TARGET_BIN": "DIABETE4_binary"}))

if group_tables:
    grouped = pd.concat(group_tables, ignore_index=True)
    to_csv(grouped, "05_grouped_numeric_by_target.csv")


# -----------------------
# -----------------------
# 7B) CLEAN BOXPLOTS FOR SLIDES
# -----------------------
import matplotlib.pyplot as plt

plot_df = focus.dropna(subset=["TARGET_BIN"]).copy()

# BMI fix: _BMI5 is BMI * 100  -> scale back to real BMI
if "_BMI5" in plot_df.columns:
    plot_df["_BMI5_real"] = plot_df["_BMI5"] / 100

def make_boxplot(df, value_col, title, ylabel, filename):
    no_diab = df.loc[df["TARGET_BIN"] == 0, value_col].dropna()
    diab    = df.loc[df["TARGET_BIN"] == 1, value_col].dropna()

    print(f"[PLOT] {value_col}: n_no={len(no_diab)}, n_yes={len(diab)}")

    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [no_diab, diab],
        labels=["No diabetes", "Diabetes"],
        showfliers=False  # removes the black dots
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()

# AGE (still coded as categories, but fine for EDA illustration)
if "_AGE80" in plot_df.columns:
    make_boxplot(plot_df, "_AGE80",
                 "Age category (_AGE80) by diabetes status",
                 "_AGE80 (age category code)",
                 "SLIDE_boxplot_AGE80_noFliers.png")

# BMI (real)
if "_BMI5_real" in plot_df.columns:
    make_boxplot(plot_df, "_BMI5_real",
                 "BMI by diabetes status",
                 "BMI",
                 "SLIDE_boxplot_BMI_real_noFliers.png")


# Crosstabs for categorical predictors (optional)
def crosstab_norm(df_in: pd.DataFrame, feature: str) -> pd.DataFrame:
    tab = pd.crosstab(df_in[feature], df_in["TARGET_BIN"], normalize="index") * 100
    tab = tab.round(3)
    tab.columns = [f"DIABETE4_binary={c}" for c in tab.columns]
    return tab

cat_predictors = ["SEXVAR", "EDUCA", "INCOME3", "_SMOKER3", "_TOTINDA", "BPHIGH6", "TOLDHI3"]
for cat_col in cat_predictors:
    if cat_col in focus.columns:
        tab = crosstab_norm(focus.dropna(subset=["TARGET_BIN"]), cat_col)
        to_csv(tab, f"05_crosstab_{cat_col}_by_target_percent.csv")


# -----------------------
# 8) STEP 6 — RISKS: SPARSITY + SUBGROUP DISTRIBUTIONS
# -----------------------
sparsity_rows = []
for col in predictors_found:
    vc = df[col].value_counts(dropna=False)
    n_unique = int(vc.shape[0])
    n_small = int((vc < 50).sum())
    sparsity_rows.append({"feature": col, "n_unique": n_unique, "n_categories_lt50": n_small})

if sparsity_rows:
    sparsity = pd.DataFrame(sparsity_rows).sort_values("n_categories_lt50", ascending=False)
    to_csv(sparsity, "06_sparsity_summary.csv")
else:
    write_text("06_sparsity_summary.txt", "No predictors found; sparsity summary not computed.")

for subgroup in ["SEXVAR", "INCOME3", "EDUCA"]:
    if subgroup in focus.columns:
        tab = pd.crosstab(focus[subgroup], focus["TARGET_BIN"], normalize="index") * 100
        tab = tab.round(3)
        to_csv(tab, f"06_target_distribution_within_{subgroup}.csv")


# -----------------------
# 9) STEP 7 — DATA LEAKAGE SCAN (COLUMN NAME SEARCH)
# -----------------------
leak_hits = []
for col in df.columns:
    for kw in LEAKAGE_KEYWORDS:
        if kw in col:
            leak_hits.append({"column": col, "keyword": kw})

leak_df = pd.DataFrame(leak_hits).drop_duplicates()
if not leak_df.empty:
    leak_df = leak_df.sort_values(["keyword", "column"])
to_csv(leak_df, "07_leakage_keyword_hits.csv")


# -----------------------
# 10) WRITE A SHORT TEXT SUMMARY
# -----------------------
summary_lines = []
summary_lines.append("EDA SUMMARY (auto-generated from LLCP2023.XPT)\n")
summary_lines.append(f"- Dataset size: {df.shape[0]} rows, {df.shape[1]} columns.")
summary_lines.append(f"- Target: {TARGET} (binary: diabetes=1 vs no diabetes=0; counts: diabetes={yes}, no={no}).")

if predictors_missing:
    summary_lines.append(f"- WARNING: Some expected predictors were not found: {predictors_missing}")
else:
    summary_lines.append("- All expected predictors were found in the dataset.")

top_miss_focus = miss_focus.sort_values("missing_rate", ascending=False).head(8)
summary_lines.append("- Highest missingness (predictors + target):")
for idx, row in top_miss_focus.iterrows():
    summary_lines.append(f"  * {idx}: {row['missing_percent']}% missing")

summary_lines.append(f"- Potential leakage-related columns flagged by keyword scan: {len(leak_df)} (see 07_leakage_keyword_hits.csv).")

sleep_like = [c for c in df.columns if ("SLEP" in c) or ("SLEEP" in c)]
summary_lines.append(f"- Sleep-duration check: columns containing 'SLEP'/'SLEEP' found: {sleep_like} (expected: none).")

summary_lines.append("\nNext step suggestion:")
summary_lines.append("- Use these EDA findings to define preprocessing rules (handling non-response codes, missingness strategy, outlier rules, and leakage exclusions).")
summary_lines.append("- Keep EDA and preprocessing separate for transparency and reproducibility.")

write_text("EDA_summary.txt", "\n".join(summary_lines))

print("\nDone. Outputs written to:", OUT_DIR.resolve())
print("Key files: EDA_summary.txt, 02_target_value_counts.csv, 02_target_binary_value_counts.csv, 03_missing_focus_predictors_target.csv, 07_leakage_keyword_hits.csv")
