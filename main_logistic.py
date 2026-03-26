"""
Úloha 2 – Logistická regrese: předpovídáme passed (0 / 1)
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

# ÚLOHA 1 – Načtení
df_raw = pd.read_csv("dataset/dataset.csv")
print("=" * 50)
print("ÚLOHA 1 – Načtení dat")
print("=" * 50)
print(df_raw.head())

# ÚLOHA 2 – Čištění
print("\n" + "=" * 50)
print("ÚLOHA 2 – Čištění dat")
print("=" * 50)
df = clean(df_raw)

# ÚLOHA 3 – Klasifikační proměnná
print("\n" + "=" * 50)
print("ÚLOHA 3 – Klasifikační proměnná 'passed'")
print("=" * 50)
df["passed"] = (df["final_score"] >= 75).astype(int)
print(df[["final_score", "passed"]].to_string(index=False))
print(f"\n  Prošlo: {df['passed'].sum()}  |  Neprošlo: {(df['passed'] == 0).sum()}")

# ÚLOHA 4 – Explorace skupin
print("\n" + "=" * 50)
print("ÚLOHA 4 – Rozdíly mezi skupinami")
print("=" * 50)
group = df.groupby("passed")[["study_hours", "sleep_hours", "attendance"]].mean()
group.index = group.index.map({0: "Neprošel", 1: "Prošel"})
print(group.round(2).to_string())

# ÚLOHA 5 – Model
print("\n" + "=" * 50)
print("ÚLOHA 5 – Model (logistická regrese)")
print("=" * 50)
features = ["study_hours", "sleep_hours", "attendance"]
X = df[features].values
y = df["passed"].values
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)

# ÚLOHA 6 – Vyhodnocení
print("\n" + "=" * 50)
print("ÚLOHA 6 – Vyhodnocení modelu")
print("=" * 50)
print(f"  Přesnost (accuracy): {accuracy_score(y, y_pred) * 100:.1f} %")
print()
print(classification_report(y, y_pred, target_names=["Neprošel", "Prošel"]))
comparison = df[["final_score", "passed"]].copy()
comparison["predicted"] = y_pred
comparison["správně"] = comparison["passed"] == comparison["predicted"]
print(comparison.to_string(index=False))

# ÚLOHA 7 – Pravděpodobnosti
print("\n" + "=" * 50)
print("ÚLOHA 7 – Pravděpodobnosti")
print("=" * 50)
proba = model.predict_proba(X)
df["prob_pass"] = proba[:, 1]
print(df[["student_id", "final_score", "passed", "prob_pass"]].round(3).to_string(index=False))

# ÚLOHA 8 – Koeficienty
print("\n" + "=" * 50)
print("ÚLOHA 8 – Koeficienty modelu")
print("=" * 50)
for feat, coef in zip(features, model.coef_[0]):
    smer = "zvyšuje" if coef > 0 else "snižuje"
    print(f"  {feat}: {coef:+.4f}  → {smer} šanci na úspěch")

# ÚLOHA 9 – Varování
print("\n" + "=" * 50)
print("ÚLOHA 9 – Varování studentů (práh 0.5)")
print("=" * 50)
df["warning"] = df["prob_pass"] < 0.5
at_risk = df[df["warning"]][["student_id", "prob_pass", "final_score"]]
if len(at_risk) > 0:
    print("  Studenti v riziku:")
    print(at_risk.round(3).to_string(index=False))
else:
    print("  Žádný student není v riziku dle modelu.")

# ÚLOHA 10 – Vlastní analýza
print("\n" + "=" * 50)
print("ÚLOHA 10 – Vlastní analýza: docházka vs pravděpodobnost")
print("=" * 50)
df["attendance_group"] = pd.cut(df["attendance"], bins=[0, 75, 100], labels=["nízká (<75%)", "vysoká (≥75%)"])
result = df.groupby("attendance_group", observed=True)["prob_pass"].mean()
print(result.round(3).to_string())