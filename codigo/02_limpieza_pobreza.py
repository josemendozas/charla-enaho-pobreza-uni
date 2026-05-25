"""
02_limpieza_pobreza.py
Limpieza de ENAHO y construcción de variables de pobreza monetaria.

Requisitos:
  - Haber corrido 01_descarga_enaho.py (o tener los .parquet en datos/procesados/)
  - pip install pandas pyarrow statsmodels matplotlib

Este script:
  1. Carga el módulo Sumaria (2021-2025)
  2. Construye variables clave de pobreza
  3. Calcula indicadores usando factores de expansión
  4. Genera un dataset ejemplo listo para análisis
  5. Muestra cómo hacer el merge con datos administrativos
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
DATA_DIR = Path("../datos/procesados")
OUT_DIR = Path("../datos")

# --------------------------------------------------------------------------- #
# 1. Cargar Sumaria
# --------------------------------------------------------------------------- #
print("=" * 60)
print("1. Cargando módulo Sumaria")
print("=" * 60)

sumaria_path = DATA_DIR / "sumaria_2021_2025.parquet"

if sumaria_path.exists():
    df = pd.read_parquet(sumaria_path)
    print(f"   Filas: {len(df):,}")
else:
    # Si no existe el parquet, intentar cargar un .dta de ejemplo
    print("   [INFO] No se encontró el parquet apilado.")
    print("   Intentando cargar un archivo .dta de ejemplo...")
    ejemplo = list(Path("../datos/raw").rglob("*sumaria*.dta"))
    if ejemplo:
        df = pd.read_stata(ejemplo[0], convert_categoricals=False)
        df.columns = df.columns.str.lower().str.strip()
        print(f"   Cargado: {ejemplo[0].name} ({len(df):,} filas)")
    else:
        print("   [ERROR] No se encontraron datos. Descarga ENAHO primero.")
        print("   Creando dataset de demostración con datos sintéticos...")

        # Dataset sintético para que los alumnos puedan correr el código
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "anio": np.random.choice(range(2021, 2026), n),
            "conglome": np.random.randint(1, 500, n).astype(str),
            "viession": np.random.randint(1, 20, n).astype(str),
            "hoession": np.random.randint(1, 5, n).astype(str),
            "ubigeo": np.random.choice(
                ["010101", "010201", "040101", "050101", "130101",
                 "150101", "150132", "200101", "210101", "250101"], n
            ),
            "domession": np.random.choice(range(1, 9), n),
            "estrato": np.random.choice(range(1, 6), n),
            "mieperho": np.random.choice(range(1, 8), n, p=[0.05, 0.15, 0.25, 0.25, 0.15, 0.1, 0.05]),
            "factor07": np.random.uniform(100, 2000, n),
            "gashog2d": np.random.lognormal(10.2, 0.9, n),  # gasto bruto hogar anual
            "linea": np.random.uniform(380, 480, n),           # línea de pobreza
            "linpe": np.random.uniform(200, 270, n),            # línea pobreza extrema
            "inghog1d": np.random.lognormal(10.5, 1.0, n),   # ingreso bruto hogar anual
            "poession": np.random.randint(0, 100, n),  # placeholder
        })
        print(f"   Dataset sintético creado: {len(df):,} filas")

print(f"   Columnas disponibles: {len(df.columns)}")

# --------------------------------------------------------------------------- #
# 2. Construir variables de pobreza
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("2. Construyendo variables de pobreza")
print("=" * 60)

# Gasto per cápita mensual
df["gasto_pc"] = df["gashog2d"] / (df["mieperho"] * 12)

# Clasificación de pobreza
df["pobre"] = (df["gasto_pc"] < df["linea"]).astype(int)
df["pobre_extremo"] = (df["gasto_pc"] < df["linpe"]).astype(int)

# Categorías
df["categoria_pobreza"] = np.where(
    df["pobre_extremo"] == 1, "Pobre extremo",
    np.where(df["pobre"] == 1, "Pobre no extremo", "No pobre")
)

# Ubigeo componentes
df["departamento"] = df["ubigeo"].astype(str).str.zfill(6).str[:2]
df["provincia"] = df["ubigeo"].astype(str).str.zfill(6).str[:4]

# Dominio geográfico
dominios = {
    1: "Costa urbana", 2: "Costa rural",
    3: "Sierra urbana", 4: "Sierra rural",
    5: "Selva urbana", 6: "Selva rural",
    7: "Lima Metropolitana", 8: "Callao"
}
if "domession" in df.columns:
    df["dominio"] = df["domession"].map(dominios)

# Factor de expansión poblacional
df["factor_pob"] = df["factor07"] * df["mieperho"]

print(f"   Variables creadas: gasto_pc, pobre, pobre_extremo, categoria_pobreza")
print(f"   Variables auxiliares: departamento, provincia, dominio, factor_pob")

# --------------------------------------------------------------------------- #
# 3. Estadísticas descriptivas con factores de expansión
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("3. Estadísticas descriptivas (ponderadas)")
print("=" * 60)


def media_ponderada(values, weights):
    """Media ponderada evitando NaN."""
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def tasa_ponderada(indicator, weights):
    """Tasa ponderada (proporción) en porcentaje."""
    return media_ponderada(indicator, weights) * 100


# Tasa de pobreza por año
print("\n   Tasa de pobreza por año:")
print("   " + "-" * 40)
for anio in sorted(df["anio"].unique()):
    sub = df[df["anio"] == anio]
    tasa = tasa_ponderada(sub["pobre"], sub["factor_pob"])
    tasa_ext = tasa_ponderada(sub["pobre_extremo"], sub["factor_pob"])
    n = len(sub)
    print(f"   {anio}: Pobreza = {tasa:.1f}%  |  Extrema = {tasa_ext:.1f}%  |  N = {n:,}")

# Tasa de pobreza por dominio (último año)
ultimo_anio = df["anio"].max()
sub = df[df["anio"] == ultimo_anio]
print(f"\n   Pobreza por dominio ({ultimo_anio}):")
print("   " + "-" * 50)
if "dominio" in sub.columns:
    for dom in sorted(sub["dominio"].dropna().unique()):
        s = sub[sub["dominio"] == dom]
        tasa = tasa_ponderada(s["pobre"], s["factor_pob"])
        print(f"   {dom:25s}: {tasa:.1f}%  (N = {len(s):,})")

# --------------------------------------------------------------------------- #
# 4. Ejemplo de merge con datos administrativos
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("4. Ejemplo: merge con datos administrativos")
print("=" * 60)

# Simulamos un dataset administrativo (en la práctica, descargarías de SBS/SIAF)
admin_ejemplo = pd.DataFrame({
    "ubigeo": ["010101", "010201", "040101", "050101", "130101",
               "150101", "150132", "200101", "210101", "250101"],
    "n_agentes_bancarios": [3, 0, 5, 1, 8, 45, 12, 2, 0, 1],
    "gasto_inversion_pc": [250, 180, 420, 310, 560, 890, 620, 200, 150, 280],
})

# Merge: ENAHO (hogar) + admin (distrito)
df_merged = df.merge(admin_ejemplo, on="ubigeo", how="left")

print(f"   Filas antes del merge: {len(df):,}")
print(f"   Filas después del merge: {len(df_merged):,}")
print(f"   Columnas nuevas: n_agentes_bancarios, gasto_inversion_pc")

# Ejemplo rápido: pobreza en distritos con vs sin agentes bancarios
tiene_agente = df_merged["n_agentes_bancarios"] > 0
tasa_con = tasa_ponderada(
    df_merged.loc[tiene_agente, "pobre"],
    df_merged.loc[tiene_agente, "factor_pob"]
)
tasa_sin = tasa_ponderada(
    df_merged.loc[~tiene_agente, "pobre"],
    df_merged.loc[~tiene_agente, "factor_pob"]
)
print(f"\n   Pobreza en distritos CON agente bancario:  {tasa_con:.1f}%")
print(f"   Pobreza en distritos SIN agente bancario:  {tasa_sin:.1f}%")
print(f"   Diferencia: {tasa_sin - tasa_con:.1f} pp")
print(f"   (¡CUIDADO! Esto es correlación, no causalidad)")

# --------------------------------------------------------------------------- #
# 5. Guardar dataset procesado
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("5. Guardando dataset procesado")
print("=" * 60)

cols_guardar = [
    "anio", "conglome", "viession", "hoession", "ubigeo",
    "departamento", "provincia", "dominio",
    "mieperho", "factor07", "factor_pob",
    "gashog2d", "gasto_pc", "linea", "linpe",
    "pobre", "pobre_extremo", "categoria_pobreza",
    "inghog1d",
]
cols_disponibles = [c for c in cols_guardar if c in df.columns]

out_path = OUT_DIR / "enaho_ejemplo.csv"
df[cols_disponibles].to_csv(out_path, index=False)
print(f"   Guardado: {out_path} ({len(df):,} filas, {len(cols_disponibles)} columnas)")

print("\n" + "=" * 60)
print("Listo. Dataset de ejemplo disponible en datos/enaho_ejemplo.csv")
print("=" * 60)
