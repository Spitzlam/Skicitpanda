# Analýza výkonu studenta – ML projekt

Projekt obsahuje dvě analýzy nad stejným datasetem:
- **Lineární regrese** (`main_linear.py`) – předpovídáme číslo (`final_score`)
- **Logistická regrese** (`main_logistic.py`) – předpovídáme kategorii (`passed` = 0/1)

## Struktura
```
main_linear.py
main_logistic.py
dataset/
  dataset.csv
README.md
```

---

## Čištění dat (Úloha 2)

Dataset obsahoval tyto typy chyb:

| Typ chyby | Příklad | Řešení |
|-----------|---------|--------|
| Chybějící hodnoty | `NaN`, prázdné buňky | Doplněno mediánem sloupce |
| Neplatné hodnoty | `"invalid"`, `"?"` | Nahrazeno `NaN`, pak mediánem |
| Špatný datový typ | sloupce jako `str` místo `float` | `pd.to_numeric(..., errors="coerce")` |
| Duplicity | stejné řádky | `drop_duplicates()` |

Medián byl zvolen místo průměru, protože je odolnější vůči odlehlým hodnotám.

---

## Část 1 – Lineární regrese

### Popis modelu
Model `LinearRegression` ze scikit-learn předpovídá číselnou hodnotu `final_score`
na základě tří vstupů: `study_hours`, `sleep_hours`, `attendance`.

### Výsledky modelu
- **MAE** (průměrná chyba): ~1.93 bodů
- **R²** (přesnost): ~0.952 → model vysvětluje 95 % variability

### Koeficienty
- `study_hours`: +1.15 → každá hodina učení navíc = +1.15 bodu
- `sleep_hours`: +2.52 → spánek má silný pozitivní vliv
- `attendance`: +0.67 → vyšší docházka = lepší výsledek

### Odpovědi na otázky

**Co má větší vliv – učení nebo spánek?**
Korelace s `final_score`: `study_hours` = 0.834, `sleep_hours` = 0.724.
Učení má mírně větší vliv, ale spánek je v modelu silnější (koeficient 2.52 vs 1.15) –
pravděpodobně proto, že spánek ovlivňuje celkovou výkonnost mozku.

**Je vztah lineární?**
Ano – korelace jsou vysoké a model dosahuje R² 0.95, což naznačuje lineární vztah.

**Pomáhá káva?**
Korelace `coffee_cups` vs `final_score` = **−0.81** (záporná). Studenti pijící více kávy
mají horší výsledky – pravděpodobně proto, že káva nahrazuje spánek, ne učení.

**Která veličina nejvíce koreluje s výsledkem?**
`previous_score` (0.998) – minulý výsledek nejlépe předpovídá budoucí.
Z ovlivnitelných faktorů: `attendance` (0.96) > `study_hours` (0.83) > `sleep_hours` (0.72).

**Vlastní analýza – kombinace učení a spánku:**
Vytvořil jsem index `study_sleep_index = study_hours × sleep_hours`.
Korelace s `final_score` = 0.895 – kombinace je silnější než každý faktor samostatně.

---

## Část 2 – Logistická regrese

### Proč kategorie, ne číslo?
Škola nepotřebuje vědět přesné skóre, ale jen **projde / neprojde**.
Logistická regrese odhaduje **pravděpodobnost** (0–1), že nastane jev (passed = 1).
To je klasifikační úloha – výstup je rozhodnutí, ne číslo.

### Klasifikační pravidlo
```python
passed = (final_score >= 75).astype(int)
```
Z 12 studentů: 8 prošlo, 4 neprošlo.

### Rozdíly mezi skupinami (Úloha 4)

| Skupina | study_hours | sleep_hours | attendance |
|---------|-------------|-------------|------------|
| Neprošel | 3.75 | 6.00 | 71.25 % |
| Prošel   | 7.00 | 7.19 | 91.00 % |

Studenti, kteří prošli, se učí téměř **dvakrát více** a mají výrazně vyšší docházku.

### Výsledky modelu
- **Accuracy: 100 %** – model správně klasifikoval všechny studenty.
- Pozor: 100 % přesnost na trénovacích datech může být přeučení (overfitting) – dataset je malý (12 řádků).

### Co znamená pravděpodobnost 0.8?
Model říká: „Tento student projde se pravděpodobností 80 %."
Standardní hranice (threshold) je 0.5 – nad ní model predikuje `passed = 1`.

### Jak bys nastavil hranici?
- Threshold 0.5 je vhodný jako výchozí.
- Pokud chceme varovat co nejvíce rizikových studentů (i za cenu falešných poplachů), snížíme threshold na 0.3.

### Koeficienty
- `attendance`: +0.71 → docházka má největší vliv
- `study_hours`: +0.41 → více učení zvyšuje šanci
- `sleep_hours`: +0.16 → spánek pomáhá, ale méně

### Varování studentů (Úloha 9)
Studenti s `prob_pass < 0.5` by měli být upozorněni:

| student_id | prob_pass | final_score |
|------------|-----------|-------------|
| 2 | 0.122 | 68 |
| 4 | 0.000 | 62 |
| 6 | 0.000 | 58 |
| 9 | 0.007 | 72 |

**V praxi:** Model by mohl automaticky označit studenty v riziku již v polovině semestru,
a škola by jim mohla nabídnout doučování.

### Vlastní analýza – docházka vs pravděpodobnost
- Nízká docházka (<75 %): průměrná `prob_pass` = **0.002**
- Vysoká docházka (≥75 %): průměrná `prob_pass` = **0.888**

Docházka je nejsilnějším prediktorem úspěchu v tomto modelu.

---

## Co se tím učíme

| Pojem | Vysvětlení |
|-------|-----------|
| Lineární regrese | Předpovídá číslo (final_score). Hledá přímku, která nejlépe popisuje data. |
| Logistická regrese | Předpovídá pravděpodobnost kategorie (passed). Výstup je 0–1. |
| MAE | Průměrná absolutní chyba – o kolik bodů se model průměrně mýlí |
| R² | Jak dobře model vysvětluje data (1.0 = perfektní) |
| Accuracy | Podíl správně klasifikovaných případů |
| Threshold | Hranice pravděpodobnosti pro rozhodnutí (standardně 0.5) |
| Overfitting | Model se „naučil" trénovací data příliš dobře – na nových datech může selhat |
