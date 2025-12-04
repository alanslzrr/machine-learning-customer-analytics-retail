"""
Script: Análisis de Valores Faltantes
=====================================

Este módulo concentra el análisis de valores faltantes previo a la imputación,
basado en el paso 1.1.5 del notebook. Trabaja sobre el dataset crudo,
genera reportes y visualizaciones, y documenta cantidades y porcentajes
de nulos por variable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Directorio de datos intermedios (sin depender de imports de paquete)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
    missing_df = (
        df.isna()
        .sum()
        .to_frame("missing")
        .assign(missing_pct=lambda x: (x["missing"] / len(df) * 100).round(2))
        .sort_values("missing", ascending=False)
    )
    return missing_df


def _plot_missing_bar(missing_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    top = missing_df[missing_df["missing"] > 0]
    if top.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        y=top.index,
        x=top["missing_pct"],
        palette="mako",
    )
    plt.xlabel("% de valores faltantes")
    plt.ylabel("Variable")
    plt.title("Valores faltantes por variable")
    plt.tight_layout()

    fig_path = output_dir / "missing_values_bar.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()


def run_missing_values_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analiza valores faltantes sobre el dataset crudo.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original sin imputar.
    output_dir : Path
        Carpeta donde se guardarán las figuras generadas por este módulo.
    """
    print("=== 1.1.5 Análisis de valores faltantes ===")

    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    missing_df = _missing_report(df)
    csv_path = DATA_INTERIM_DIR / "eda_missing_detailed.csv"
    missing_df.to_csv(csv_path)

    print("Resumen de nulos por columna (top 15):")
    print(missing_df.head(15))
    print(f"Reporte completo de nulos guardado en: {csv_path}")

    _plot_missing_bar(missing_df, output_dir)


def main() -> None:
    """
    Punto de entrada opcional para ejecutar solo este módulo.
    """
    import main_eda  # type: ignore

    df_raw = main_eda.load_data()
    figures_dir = Path(__file__).resolve().parent / "figures" / "missing"
    run_missing_values_analysis(df_raw, figures_dir)


if __name__ == "__main__":
    main()
