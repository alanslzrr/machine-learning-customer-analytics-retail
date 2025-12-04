"""
Script: Análisis de Vista General del Dataset
============================================

OBJETIVO
--------
Proporcionar una vista estructural inicial del dataset original, similar al
paso 1.1 del notebook `proyecto_00_eda[1].ipynb`:

- Dimensiones y primeras filas.
- Tipos de datos.
- Resumen de valores faltantes y duplicados.
- Cardinalidad de variables categóricas.
- Validaciones básicas de integridad.

Este módulo NO modifica el dataset; sólo genera diagnósticos y gráficos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Directorio de datos intermedios (sin depender de imports de paquete)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def _save_missing_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Calcula y guarda un resumen de valores faltantes por columna.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_df = (
        df.isna()
        .sum()
        .to_frame("missing")
        .assign(missing_pct=lambda x: (x["missing"] / len(df) * 100).round(2))
        .sort_values("missing", ascending=False)
    )

    csv_path = DATA_INTERIM_DIR / "eda_missing_summary.csv"
    missing_df.to_csv(csv_path)

    return missing_df


def _plot_categorical_cardinality(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Genera un gráfico de barras con la cardinalidad de las variables categóricas.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_cols = df.select_dtypes(include="object").nunique().sort_values(ascending=False)
    if cat_cols.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_cols.values, y=cat_cols.index, palette="viridis")
    plt.xlabel("Número de valores únicos")
    plt.ylabel("Variable categórica")
    plt.title("Cardinalidad de variables categóricas")
    plt.tight_layout()

    fig_path = output_dir / "categorical_cardinality.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()


def _validate_primary_key_and_constants(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """
    Replica la validación de unicidad de clave primaria y columnas constantes.
    """
    pk_unique = False
    if "id" in df.columns:
        pk_unique = df["id"].is_unique

    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    return pk_unique, const_cols


def run_data_overview(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Ejecuta el análisis estructural inicial sobre un DataFrame ya cargado.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original sin transformar.
    output_dir : Path
        Carpeta donde se guardarán las figuras generadas por este módulo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1.1 Vista general del dataset ===")
    print(f"Shape: {df.shape}")
    print(f"Número de columnas: {len(df.columns)}")

    # Tipos de datos
    dtypes_path = DATA_INTERIM_DIR / "eda_dtypes.csv"
    df.dtypes.sort_index().to_frame("dtype").to_csv(dtypes_path)
    print(f"Tipos de datos guardados en: {dtypes_path}")

    # Resumen de missing y duplicados
    missing_df = _save_missing_summary(df, output_dir)
    dup_total = df.duplicated().sum()
    print("\nTop 10 columnas con más nulos:")
    print(missing_df.head(10))
    print(f"\nRegistros duplicados: {dup_total}")

    # Cardinalidad de variables categóricas (gráfico)
    _plot_categorical_cardinality(df, output_dir)

    # Validaciones básicas
    pk_unique, const_cols = _validate_primary_key_and_constants(df)
    print(f"\nClave primaria única (id): {pk_unique}")
    print(f"Columnas constantes detectadas (pre-imputación): {const_cols}")

    # Validación de importes negativos en columnas de gasto
    gasto_cols = [c for c in df.columns if c.startswith("gasto_")]
    if gasto_cols:
        negatives = (df[gasto_cols] < 0).any()
        print("\nValores negativos por columna de gasto:")
        print(negatives)


def main() -> None:
    """
    Punto de entrada opcional para ejecutar solo la vista general.

    Normalmente este módulo será invocado desde `main_eda.run_eda_pipeline`
    pasando el DataFrame ya cargado.
    """
    import main_eda  # type: ignore

    df = main_eda.load_data()
    figures_dir = Path(__file__).resolve().parent / "figures" / "overview"
    run_data_overview(df, figures_dir)


if __name__ == "__main__":
    main()
