# Pobreza Monetaria y la ENAHO: Herramientas para temas de tesis

Materiales de la charla para estudiantes de pregrado de la UNI (Universidad Nacional de Ingeniería) sobre cómo usar la Encuesta Nacional de Hogares (ENAHO) para investigación en pobreza y desarrollo.

## Contenido

```
slides/charla.qmd          # Slides (Quarto/Revealjs)
codigo/01_descarga_enaho.py # Descarga y carga de microdatos ENAHO 2021-2025
codigo/02_limpieza_pobreza.py # Limpieza y construcción de variables de pobreza
datos/                      # Dataset de ejemplo (generado por el script)
referencias/papers_sugeridos.md # Papers, libros y recursos
```

## Estructura de la charla (~60 min)

1. **La ENAHO por dentro** — Diseño muestral, módulos, línea de pobreza, trampas comunes
2. **De correlación a causalidad** — DiD, RDD, IV, panel con efectos fijos
3. **Caso: agentes bancarios y remesas** — Ejemplo real de DiD escalonado con datos SBS + ENAHO
4. **ENAHO + datos administrativos** — Fuentes admin (SIAF, SBS, RENAMU) y cómo hacer el merge
5. **Menú de temas de tesis** — 10 ideas concretas con estrategia de identificación
6. **Cierre práctico** — Herramientas, flujo de trabajo, recursos

## Requisitos

```bash
# Para ver las slides
brew install --cask quarto
quarto preview slides/charla.qmd

# Para correr los scripts de Python
pip install pandas pyarrow statsmodels numpy
```

## Datos

Los microdatos ENAHO se descargan de [INEI Microdatos](https://proyectos.inei.gob.pe/microdatos/). El script `02_limpieza_pobreza.py` genera un dataset de demostración si no se tienen los datos reales.

## Autor

José Mendoza
