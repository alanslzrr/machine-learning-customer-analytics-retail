"""
Script: Análisis de Estadísticas Descriptivas
=============================================

Replica y amplía las secciones de estadísticas descriptivas y parte del
análisis visual univariado del notebook, trabajando sobre el dataset
ENRIQUECIDO (`supermercado_features.csv`).

Genera:
- Estadísticos descriptivos de variables numéricas y categóricas.
- Distribuciones univariadas de variables clave (edad, ingresos, etc.).
- Resúmenes guardados en `data/interim/` y gráficos en `scripts/01_EDA/figures/descriptive/`.
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


def _compute_and_save_descriptive_tables(df: pd.DataFrame) -> None:
    """
    Calcula estadísticas descriptivas numéricas y categóricas y las guarda.
    """
    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    if len(num_cols) > 0:
        desc_num = df[num_cols].describe().T
        desc_num_path = DATA_INTERIM_DIR / "eda_descriptive_numeric.csv"
        desc_num.to_csv(desc_num_path)
        print(f"Estadísticos numéricos guardados en: {desc_num_path}")

    if len(cat_cols) > 0:
        desc_cat = (
            df[cat_cols]
            .agg(["count", lambda s: s.nunique(), lambda s: s.mode().iloc[0] if not s.mode().empty else None])
            .T
        )
        desc_cat.columns = ["count", "nunique", "mode"]
        desc_cat_path = DATA_INTERIM_DIR / "eda_descriptive_categorical.csv"
        desc_cat.to_csv(desc_cat_path)
        print(f"Estadísticos categóricos guardados en: {desc_cat_path}")


def _plot_univariate_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Genera distribuciones univariadas de variables clave, similar a 1.4.2.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = [
        "edad",
        "ingresos",
        "antiguedad_anios",
        "gasto_total",
        "ticket_promedio",
        "compras_totales",
    ]

    variables = [v for v in variables if v in df.columns]
    if not variables:
        return

    n = len(variables)
    n_rows = 2
    n_cols = int(np.ceil(n / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    fig.suptitle(
        "Distribuciones Univariadas: Variables Demográficas y Financieras",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    for idx, var in enumerate(variables):
        ax = axes[idx // n_cols, idx % n_cols]

        ax.hist(df[var], bins=50, alpha=0.6, color="steelblue", edgecolor="black", density=True)
        df[var].plot(kind="kde", ax=ax, color="darkred", linewidth=2, secondary_y=False)

        media = df[var].mean()
        mediana = df[var].median()

        ax.axvline(media, color="red", linestyle="--", linewidth=2, label=f"Media: {media:.1f}")
        ax.axvline(mediana, color="green", linestyle="--", linewidth=2, label=f"Mediana: {mediana:.1f}")

        ax.set_xlabel(var.replace("_", " ").title(), fontweight="bold")
        ax.set_ylabel("Densidad", fontweight="bold")
        ax.set_title(f"Distribución de {var.replace('_', ' ').title()}", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # Eliminar ejes sobrantes si hay
    for j in range(len(variables), n_rows * n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])

    plt.tight_layout()

    fig_path = output_dir / "univariate_distributions.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()


def run_descriptive_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Ejecuta el análisis descriptivo sobre el dataset enriquecido.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset enriquecido (`supermercado_features.csv`).
    output_dir : Path
        Carpeta donde se guardarán las figuras generadas por este módulo.
    """
    print("=== 1.4 Estadísticas descriptivas y distribuciones ===")
    _compute_and_save_descriptive_tables(df)
    _plot_univariate_distributions(df, output_dir)


def main() -> None:
    """
    Punto de entrada opcional para ejecutar solo este módulo.
    """
    import main_eda  # type: ignore

    df_raw = main_eda.load_data()
    _, df_features = main_eda._apply_minimal_cleaning_and_feature_engineering(df_raw)  # type: ignore

    figures_dir = Path(__file__).resolve().parent / "figures" / "descriptive"
    run_descriptive_statistics(df_features, figures_dir)


if __name__ == "__main__":
    main()
