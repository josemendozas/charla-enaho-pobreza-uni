"""
03_ejemplos_enaho.py
Ejemplos prácticos con ENAHO: replicar pobreza, perfil, regresiones, Gini.

Requisitos:
  - Sumaria 2024 descargada de INEI (o usar datos sintéticos del script 02)
  - pip install pandas numpy statsmodels
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
from pathlib import Path

# --------------------------------------------------------------------------- #
# Cargar datos
# --------------------------------------------------------------------------- #
DATA_DIR = Path("../datos")

# Intentar cargar datos reales, si no usar el CSV de ejemplo
sumaria_real = list(Path("../datos/raw").rglob("*sumaria*.dta"))
if sumaria_real:
    print("Cargando Sumaria real...")
    df = pd.read_stata(sumaria_real[0], convert_categoricals=False)
    df.columns = df.columns.str.lower().str.strip()
else:
    print("Usando dataset de ejemplo (sintético)...")
    df = pd.read_csv(DATA_DIR / "enaho_ejemplo.csv")

print(f"Filas: {len(df):,}")

# --------------------------------------------------------------------------- #
# 1. Replicar la tasa de pobreza
# --------------------------------------------------------------------------- #
print("\n" + "=" * 50)
print("1. REPLICANDO TASA DE POBREZA")
print("=" * 50)

# Gasto per cápita mensual
if "gasto_pc" not in df.columns:
    df["gasto_pc"] = df["gashog2d"] / (df["mieperho"] * 12)

# Clasificar pobreza
df["pobre"] = (df["gasto_pc"] < df["linea"]).astype(int)
df["pobre_extremo"] = (df["gasto_pc"] < df["linpe"]).astype(int)

# Factor de expansión poblacional
df["factor_pob"] = df["factor07"] * df["mieperho"]

# Tasa ponderada
tasa = np.average(df["pobre"], weights=df["factor_pob"]) * 100
tasa_ext = np.average(df["pobre_extremo"], weights=df["factor_pob"]) * 100
print(f"  Pobreza: {tasa:.1f}%")
print(f"  Pobreza extrema: {tasa_ext:.1f}%")
print("  (Con datos reales de 2024 debería dar 27.6% y 5.5%)")

# --------------------------------------------------------------------------- #
# 2. Perfil del pobre
# --------------------------------------------------------------------------- #
print("\n" + "=" * 50)
print("2. PERFIL DEL POBRE")
print("=" * 50)

# Por dominio geográfico
if "dominio" in df.columns:
    print("\n  Pobreza por dominio:")
    for dom in sorted(df["dominio"].dropna().unique()):
        sub = df[df["dominio"] == dom]
        mask = sub["factor_pob"].notna() & (sub["factor_pob"] > 0)
        if mask.sum() > 0:
            t = np.average(sub.loc[mask, "pobre"], weights=sub.loc[mask, "factor_pob"]) * 100
            print(f"    {dom:25s}: {t:.1f}%")

# Por tamaño de hogar
print("\n  Pobreza por tamaño de hogar:")
for size in [1, 2, 3, 4, 5]:
    if size < 5:
        sub = df[df["mieperho"] == size]
        label = f"{size} miembros"
    else:
        sub = df[df["mieperho"] >= size]
        label = f"{size}+ miembros"
    if len(sub) > 10:
        mask = sub["factor_pob"].notna() & (sub["factor_pob"] > 0)
        t = np.average(sub.loc[mask, "pobre"], weights=sub.loc[mask, "factor_pob"]) * 100
        print(f"    {label:15s}: {t:.1f}%  (N={len(sub):,})")

# --------------------------------------------------------------------------- #
# 3. Regresión: correlatos de pobreza
# --------------------------------------------------------------------------- #
print("\n" + "=" * 50)
print("3. REGRESIÓN: CORRELATOS DE POBREZA")
print("=" * 50)

# Crear variable rural si no existe
if "estrato" in df.columns:
    df["rural"] = (df["estrato"] > 5).astype(int)
elif "dominio" in df.columns:
    df["rural"] = df["dominio"].str.contains("rural", case=False, na=False).astype(int)

# Variables disponibles para regresión
vars_disponibles = []
for v in ["rural", "mieperho"]:
    if v in df.columns:
        vars_disponibles.append(v)

if vars_disponibles:
    X = df[vars_disponibles].copy()
    X = sm.add_constant(X)
    y = df["pobre"]
    w = df["factor_pob"]

    # Filtrar NaN
    mask = X.notna().all(axis=1) & y.notna() & w.notna() & (w > 0)
    X, y, w = X[mask], y[mask], w[mask]

    modelo = sm.WLS(y, X, weights=w).fit()

    print("\n  Modelo: pobre ~ rural + mieperho (WLS con factor de expansión)")
    print(f"  N = {int(modelo.nobs):,}, R² = {modelo.rsquared:.3f}")
    print("\n  Coeficientes:")
    for var in modelo.params.index:
        coef = modelo.params[var]
        se = modelo.bse[var]
        pval = modelo.pvalues[var]
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"    {var:15s}: {coef:+.4f} (SE={se:.4f}) {sig}")

    print("\n  NOTA: Esto es correlación, NO causalidad.")
else:
    print("  Variables insuficientes para regresión.")

# --------------------------------------------------------------------------- #
# 4. Desigualdad: Gini y ratio p90/p10
# --------------------------------------------------------------------------- #
print("\n" + "=" * 50)
print("4. DESIGUALDAD")
print("=" * 50)


def gini_ponderado(valores, pesos):
    """Calcula el coeficiente de Gini ponderado."""
    mask = np.isfinite(valores) & np.isfinite(pesos) & (pesos > 0) & (valores > 0)
    v, w = valores[mask], pesos[mask]
    orden = np.argsort(v)
    v, w = v[orden], w[orden]
    w_cum = np.cumsum(w) / np.sum(w)
    vw_cum = np.cumsum(v * w) / np.sum(v * w)
    return 1 - 2 * np.trapz(vw_cum, w_cum)


g = gini_ponderado(df["gasto_pc"].values, df["factor_pob"].values)
print(f"  Gini del gasto per cápita: {g:.3f}")

# Percentiles
stats = DescrStatsW(
    df.loc[df["gasto_pc"] > 0, "gasto_pc"],
    weights=df.loc[df["gasto_pc"] > 0, "factor_pob"]
)
try:
    p10 = stats.quantile(0.1).iloc[0]
    p50 = stats.quantile(0.5).iloc[0]
    p90 = stats.quantile(0.9).iloc[0]
    print(f"  P10: S/ {p10:.0f}")
    print(f"  P50 (mediana): S/ {p50:.0f}")
    print(f"  P90: S/ {p90:.0f}")
    print(f"  Ratio P90/P10: {p90/p10:.1f}x")
except Exception:
    print("  (No se pudieron calcular percentiles)")

print("\n" + "=" * 50)
print("Listo. Todo con Sumaria + factor07 + Python.")
print("=" * 50)
