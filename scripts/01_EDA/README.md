EDA – Análisis Exploratorio de Datos
===================================

Objetivo
--------

Este paquete implementa en Python, de forma modular, el flujo completo del
notebook `proyecto_00_eda[1].ipynb`:

- Carga del dataset original del supermercado.
- Limpieza mínima e imputación.
- Ingeniería de características (demográficas, gasto, comportamiento, hogar).
- Estadísticas descriptivas.
- Análisis de valores faltantes.
- Análisis exploratorio visual avanzado.


Estructura de módulos
---------------------

- `00_eda_config.py`  
  Configuración común del EDA:
  - Carga `config/config.yaml`.
  - Expone rutas a datos (`data/raw/`, `data/interim/`, etc.).
  - Define la carpeta base de figuras: `scripts/01_EDA/figures/`.

- `main_eda.py`  
  Orquestador del pipeline completo:
  - `load_data()`: carga `proy_supermercado_dev.csv` desde `data/raw/` o, en su defecto, desde la raíz del proyecto.
  - `_apply_minimal_cleaning_and_feature_engineering(df)`: replica los pasos 1.2 y 1.3 del notebook (limpieza + features).
  - `run_eda_pipeline()`: ejecuta todo el flujo EDA e invoca al resto de módulos.

- `01_data_overview.py`  
  - `run_data_overview(df, output_dir)`:
    - Dimensiones y tipos de datos.
    - Resumen de nulos y duplicados.
    - Cardinalidad de variables categóricas.
    - Validación de clave primaria y columnas constantes.
    - Verificación de valores negativos en columnas de gasto.
  - Genera CSVs de apoyo en `data/interim/` y figuras en `figures/overview/`.

- `02_descriptive_statistics.py`  
  - `run_descriptive_statistics(df, output_dir)`:
    - Estadísticos descriptivos de variables numéricas y categóricas.
    - Distribuciones univariadas de variables clave (edad, ingresos, gasto, etc.).
  - Trabaja sobre el dataset enriquecido y guarda:
    - Tablas en `data/interim/`.
    - Gráficos en `figures/descriptive/`.

- `03_missing_values_analysis.py`  
  - `run_missing_values_analysis(df, output_dir)`:
    - Resumen detallado de valores faltantes por variable (conteo y %).
    - Gráfico de barras de nulos por variable.
  - Se ejecuta sobre el dataset crudo y guarda:
    - Reportes en `data/interim/`.
    - Figuras en `figures/missing/`.

- `04_additional_eda_plots.py`  
  - `run_additional_eda_plots(df, output_dir)`:
    - Análisis de la variable objetivo (`respuesta`).
    - Matriz de correlación de variables numéricas clave.
    - Boxplots para detección visual de outliers.
    - Tasas de respuesta por variables categóricas (educación, estado civil).
    - Análisis bivariado numérico vs `respuesta` con pruebas Mann-Whitney.
  - Genera figuras en subcarpetas de `figures/additional/`.


Carpetas de salida
------------------

- `data/interim/`  
  - `supermercado_limpio.csv`: dataset limpio (tras limpieza mínima).  
  - `supermercado_features.csv`: dataset enriquecido con todas las features.  
  - `eda_dtypes.csv`, `eda_missing_summary.csv`, `eda_missing_detailed.csv`,  
    `eda_descriptive_numeric.csv`, `eda_descriptive_categorical.csv`, etc.

- `scripts/01_EDA/figures/`  
  - `overview/`: gráficos estructurales (cardinalidad, etc.).  
  - `missing/`: gráficos de nulos.  
  - `descriptive/`: distribuciones univariadas.  
  - `additional/`: análisis avanzado (target, correlaciones, outliers, bivariado).


Ejecución del EDA completo
--------------------------

Desde la raíz del proyecto:

```bash
python -m scripts.01_EDA.main_eda
```

o bien:

```bash
python scripts/01_EDA/main_eda.py
```

Esto ejecuta:

1. Carga del dataset original.
2. Vista general (`run_data_overview`).
3. Análisis de missing (`run_missing_values_analysis`).
4. Limpieza mínima e ingeniería de características.
5. Guardado de `supermercado_limpio.csv` y `supermercado_features.csv`.
6. Estadísticos descriptivos (`run_descriptive_statistics`).
7. Gráficos avanzados (`run_additional_eda_plots`).


Uso desde un `main.py` general
------------------------------

En un pipeline global del proyecto, puedes reutilizar el EDA así:

```python
from scripts.01_EDA.main_eda import run_eda_pipeline

def main():
    # Otros pasos del proyecto...
    run_eda_pipeline()
    # Continuar con preprocesamiento/modelado...

if __name__ == "__main__":
    main()
```


Notas
-----

- El flujo está alineado con las secciones 1.1–1.5 del notebook de EDA.
- Los gráficos no se muestran en pantalla; sólo se exportan a archivos PNG.
- El código está pensado para ser reproducible y fácil de integrar en pipelines
  de aprendizaje automático posteriores.


