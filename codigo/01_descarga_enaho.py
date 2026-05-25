"""
01_descarga_enaho.py
Descarga y carga de microdatos ENAHO (2021-2025)

NOTA: INEI no ofrece API pública. Los microdatos se descargan manualmente desde:
  https://proyectos.inei.gob.pe/microdatos/
  Seleccionar: Condiciones de Vida y Pobreza - ENAHO > Año > Módulos

Este script:
  1. Muestra cómo organizar los archivos descargados
  2. Carga los módulos principales en DataFrames
  3. Construye un panel apilado 2021-2025
"""

import os
from pathlib import Path
import pandas as pd
import zipfile

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
ANIOS = range(2021, 2026)  # 2021 a 2025

# Carpeta donde guardarás los .zip o .dta descargados del INEI
RAW_DIR = Path("../datos/raw")
OUT_DIR = Path("../datos/procesados")

# Módulos que vamos a usar (nombres de archivo típicos del INEI)
MODULOS = {
    "sumaria": "sumaria",       # Variables calculadas (gasto, pobreza)
    "enaho01": "enaho01",       # Vivienda y hogar (módulo 100)
    "enaho01a": "enaho01a",     # Miembros del hogar (módulo 200)
    "enaho03": "enaho03",       # Educación (módulo 300)
    "enaho04": "enaho04",       # Salud (módulo 400)
    "enaho05": "enaho05",       # Empleo e ingresos (módulo 500)
}

# --------------------------------------------------------------------------- #
# Funciones auxiliares
# --------------------------------------------------------------------------- #

def buscar_archivo(directorio: Path, patron: str) -> Path | None:
    """Busca un archivo .dta o .sav que contenga el patrón en su nombre."""
    for ext in ["*.dta", "*.DTA", "*.sav", "*.SAV"]:
        for f in directorio.rglob(ext):
            if patron.lower() in f.stem.lower():
                return f
    return None


def cargar_modulo(directorio: Path, modulo: str, anio: int) -> pd.DataFrame | None:
    """
    Carga un módulo ENAHO desde .dta o .sav.
    Busca recursivamente en la carpeta del año.
    """
    carpeta_anio = directorio / str(anio)
    if not carpeta_anio.exists():
        print(f"  [WARN] No existe carpeta {carpeta_anio}")
        return None

    archivo = buscar_archivo(carpeta_anio, modulo)
    if archivo is None:
        print(f"  [WARN] No se encontró {modulo} en {carpeta_anio}")
        return None

    print(f"  Cargando: {archivo.name}")
    if archivo.suffix.lower() == ".dta":
        df = pd.read_stata(archivo, convert_categoricals=False)
    else:
        df = pd.read_spss(archivo)

    # Normalizar nombres de columnas a minúsculas
    df.columns = df.columns.str.lower().str.strip()

    # Agregar año si no existe
    if "anio" not in df.columns and "año" not in df.columns:
        df["anio"] = anio

    return df


def extraer_zips(directorio: Path):
    """Extrae todos los .zip encontrados en el directorio."""
    for zf in directorio.rglob("*.zip"):
        destino = zf.parent / zf.stem
        if not destino.exists():
            print(f"  Extrayendo: {zf.name}")
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(destino)


# --------------------------------------------------------------------------- #
# Ejecución principal
# --------------------------------------------------------------------------- #

def main():
    # Crear directorios si no existen
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Paso 0: Extraer zips si los hay
    print("=" * 60)
    print("Paso 0: Extrayendo archivos .zip")
    print("=" * 60)
    extraer_zips(RAW_DIR)

    # Paso 1: Cargar cada módulo por año y apilar
    for nombre, patron in MODULOS.items():
        print("=" * 60)
        print(f"Cargando módulo: {nombre}")
        print("=" * 60)

        frames = []
        for anio in ANIOS:
            print(f"\n  Año {anio}:")
            df = cargar_modulo(RAW_DIR, patron, anio)
            if df is not None:
                frames.append(df)

        if frames:
            panel = pd.concat(frames, ignore_index=True)
            out_path = OUT_DIR / f"{nombre}_2021_2025.parquet"
            panel.to_parquet(out_path, index=False)
            print(f"\n  -> Guardado: {out_path} ({len(panel):,} filas)")
        else:
            print(f"\n  [ERROR] No se encontraron datos para {nombre}")

    print("\n" + "=" * 60)
    print("Descarga completa.")
    print(f"Archivos procesados en: {OUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
