"""
Úloha 1 – Lineární regrese: předpovídáme final_score (číslo)
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def clean(df):
    df = df.copy()
    df.replace(["?", "invalid", ""], pd.NA, inplace=True)
    numeric_cols = ["study_hours", "sleep_hours", "attendance",
                    "previous_score", "coffee_cups", "final_score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["attendance"] > 100, "attendance"] = pd.NA
    df.loc[df["attendance"] < 0,   "attendance"] = pd.NA
    for col in numeric_cols:
        median = df[col].median()
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  '{col}': {missing} chybějící → doplněno mediánem ({median:.1f})")
        df[col] = df[col].fillna(median)
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicity: {before - len(df)} odstraněno | Čistý dataset: {len(df)} řádků")
    return df

# ÚLOHA 1 – Načtení a první analýza
df_raw = pd.read_csv("dataset/dataset.csv")
print("=" * 50)
print("ÚLOHA 1 – Načtení dat")
print("=" * 50)
print(df_raw.head())
print()
df_raw.info()
print(f"\nPočet řádků: {df_raw.shape[0]}, sloupců: {df_raw.shape[1]}")

# ÚLOHA 2 – Čištění dat
print("\n" + "=" * 50)
print("ÚLOHA 2 – Čištění dat")
print("=" * 50)
df = clean(df_raw)

# ÚLOHA 3 – Vztahy mezi veličinami
print("\n" + "=" * 50)
print("ÚLOHA 3 – Vztahy s final_score")
print("=" * 50)
for col in ["study_hours", "sleep_hours"]:
    corr = df[col].corr(df["final_score"])
    print(f"  Korelace {col} vs final_score: {corr:.3f}")

# ÚLOHA 4 – Vliv docházky
print("\n" + "=" * 50)
print("ÚLOHA 4 – Vliv docházky")
print("=" * 50)
df["attendance_group"] = pd.cut(df["attendance"], bins=[0, 75, 100], labels=["nízká", "vysoká"])
print(df.groupby("attendance_group", observed=True)["final_score"].mean().to_string())

# ÚLOHA 5 – Vliv kávy
print("\n" + "=" * 50)
print("ÚLOHA 5 – Vliv kávy")
print("=" * 50)
corr_coffee = df["coffee_cups"].corr(df["final_score"])
print(f"  Korelace coffee_cups vs final_score: {corr_coffee:.3f}")
print("  (záporná korelace → více kávy nesouvisí s lepším výsledkem)")

# ÚLOHA 6 – Korelace
print("\n" + "=" * 50)
print("ÚLOHA 6 – Korelační matice")
print("=" * 50)
numeric_cols = ["study_hours", "sleep_hours", "attendance",
                "previous_score", "coffee_cups", "final_score"]
print(df[numeric_cols].corr()["final_score"].sort_values(ascending=False).to_string())

# ÚLOHA 7 – Model
print("\n" + "=" * 50)
print("ÚLOHA 7 – Model (lineární regrese)")
print("=" * 50)
features = ["study_hours", "sleep_hours", "attendance"]
X = df[features].values
y = df["final_score"].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print(f"  MAE (průměrná chyba): {mean_absolute_error(y, y_pred):.2f} bodů")
print(f"  R²  (přesnost modelu): {r2_score(y, y_pred):.3f}  (1.0 = perfektní)")

# ÚLOHA 8 – Koeficienty
print("\n" + "=" * 50)
print("ÚLOHA 8 – Interpretace koeficientů")
print("=" * 50)
for feat, coef in zip(features, model.coef_):
    print(f"  {feat}: {coef:+.3f}")
print(f"  Intercept: {model.intercept_:.2f}")

# ÚLOHA 9 – Vlastní analýza
print("\n" + "=" * 50)
print("ÚLOHA 9 – Vlastní analýza: index (study × sleep)")
print("=" * 50)
df["study_sleep_index"] = df["study_hours"] * df["sleep_hours"]
corr_index = df["study_sleep_index"].corr(df["final_score"])
print(f"  Korelace (study × sleep) vs final_score: {corr_index:.3f}")
print(df[["study_hours", "sleep_hours", "study_sleep_index", "final_score"]].to_string(index=False))