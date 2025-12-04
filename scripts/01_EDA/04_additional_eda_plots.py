"""
Script: Gráficos avanzados de EDA
=================================

Este módulo agrupa las secciones visuales avanzadas del notebook:
- Análisis de la variable objetivo.
- Matriz de correlación.
- Detección visual de outliers.
- Análisis categórico vs respuesta.
- Análisis bivariado de variables numéricas clave vs respuesta.

Trabaja sobre el dataset enriquecido (`supermercado_features.csv`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


COLORES_RESPUESTA = {0: "#e74c3c", 1: "#2ecc71"}  # Rojo: No, Verde: Sí


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_target_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    if "respuesta" not in df.columns:
        return

    output_dir = _ensure_dir(output_dir / "target")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Análisis de la Variable Objetivo: Respuesta a la Campaña", fontsize=16, fontweight="bold")

    # 1. Distribución de la variable objetivo
    ax1 = axes[0]
    conteo = df["respuesta"].value_counts()
    porcentajes = df["respuesta"].value_counts(normalize=True) * 100

    bars = ax1.bar(
        [0, 1],
        conteo,
        color=[COLORES_RESPUESTA[0], COLORES_RESPUESTA[1]],
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )

    for bar, val, pct in zip(bars, conteo, porcentajes):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax1.set_xlabel("Respuesta", fontweight="bold")
    ax1.set_ylabel("Frecuencia", fontweight="bold")
    ax1.set_title("Distribución de Respuestas", fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["No (0)", "Sí (1)"])
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Gasto total por respuesta
    ax2 = axes[1]
    if "gasto_total" in df.columns:
        df.boxplot(
            column="gasto_total",
            by="respuesta",
            ax=ax2,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )
        ax2.set_xlabel("Respuesta", fontweight="bold")
        ax2.set_ylabel("Gasto Total", fontweight="bold")
        ax2.set_title("Gasto Total por Respuesta", fontweight="bold")
        ax2.set_xticklabels(["No (0)", "Sí (1)"])

    # 3. Ingresos por respuesta
    ax3 = axes[2]
    if "ingresos" in df.columns:
        df.boxplot(
            column="ingresos",
            by="respuesta",
            ax=ax3,
            patch_artist=True,
            boxprops=dict(facecolor="lightcoral", alpha=0.7),
            medianprops=dict(color="darkred", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )
        ax3.set_xlabel("Respuesta", fontweight="bold")
        ax3.set_ylabel("Ingresos", fontweight="bold")
        ax3.set_title("Ingresos por Respuesta", fontweight="bold")
        ax3.set_xticklabels(["No (0)", "Sí (1)"])

    plt.suptitle("")
    plt.tight_layout()

    fig_path = output_dir / "target_overview.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()


def _plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir = _ensure_dir(output_dir / "correlation")

    vars_numericas = [
        "edad",
        "ingresos",
        "antiguedad_anios",
        "recencia",
        "gasto_total",
        "categorias_compradas",
        "compras_totales",
        "compras_online",
        "tasa_compra_online",
        "ticket_promedio",
        "tamano_hogar",
        "total_dependientes",
        "respuesta",
    ]
    vars_numericas = [v for v in vars_numericas if v in df.columns]
    if not vars_numericas:
        return

    corr_matrix = df[vars_numericas].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlación de Pearson"},
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    ax.set_title("Matriz de Correlación: Variables Numéricas Clave", fontsize=16, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig_path = output_dir / "correlation_heatmap.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()


def _plot_outliers_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir = _ensure_dir(output_dir / "outliers")

    variables_analizar = ["edad", "ingresos", "gasto_total", "ticket_promedio", "compras_totales"]
    variables_analizar = [v for v in variables_analizar if v in df.columns]
    if not variables_analizar:
        return

    n = len(variables_analizar)
    n_rows = 2
    n_cols = int(np.ceil(n / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    fig.suptitle("Detección Visual de Outliers (Boxplots)", fontsize=16, fontweight="bold")

    for idx, var in enumerate(variables_analizar):
        ax = axes[idx // n_cols, idx % n_cols]
        bp = ax.boxplot(
            df[var],
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor="red", markersize=4, alpha=0.5),
        )
        ax.set_ylabel(var.replace("_", " ").title(), fontweight="bold")
        ax.set_title(var.replace("_", " ").title(), fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    for j in range(len(variables_analizar), n_rows * n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])

    plt.tight_layout()
    fig_path = output_dir / "outliers_boxplots.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()


def _plot_categorical_vs_target(df: pd.DataFrame, output_dir: Path) -> None:
    if "respuesta" not in df.columns:
        return

    output_dir = _ensure_dir(output_dir / "categorical_vs_target")
    cat_vars = [v for v in ["educacion", "estado_civil"] if v in df.columns]
    if not cat_vars:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Tasas de Respuesta por Variables Categóricas", fontsize=16, fontweight="bold", y=0.995)

    for idx, var in enumerate(cat_vars):
        ax = axes[idx]
        tasa_respuesta = (
            df.groupby(var)["respuesta"]
            .agg(
                [
                    ("total", "count"),
                    ("positivos", "sum"),
                    ("tasa", "mean"),
                ]
            )
            .sort_values("tasa", ascending=False)
        )

        x = range(len(tasa_respuesta))
        width = 0.35
        ax2 = ax.twinx()

        bars1 = ax.bar(
            [i - width / 2 for i in x],
            tasa_respuesta["total"],
            width,
            label="Total clientes",
            color="steelblue",
            alpha=0.7,
        )

        ax2.plot(
            x,
            tasa_respuesta["tasa"] * 100,
            "o-",
            color="darkred",
            linewidth=2.5,
            markersize=8,
            label="Tasa de respuesta (%)",
        )

        for i, (val, tasa) in enumerate(zip(tasa_respuesta["total"], tasa_respuesta["tasa"] * 100)):
            ax.text(i - width / 2, val + 5, f"{int(val)}", ha="center", va="bottom", fontsize=9)
            ax2.text(
                i,
                tasa + 1,
                f"{tasa:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="darkred",
            )

        ax.set_xlabel(var.replace("_", " ").title(), fontweight="bold", fontsize=11)
        ax.set_ylabel("Número de clientes", fontweight="bold", fontsize=10, color="steelblue")
        ax2.set_ylabel("Tasa de respuesta (%)", fontweight="bold", fontsize=10, color="darkred")
        ax.set_xticks(list(x))
        ax.set_xticklabels(tasa_respuesta.index, rotation=45, ha="right")
        ax.tick_params(axis="y", labelcolor="steelblue")
        ax2.tick_params(axis="y", labelcolor="darkred")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(f"Respuesta por {var.replace('_', ' ').title()}", fontweight="bold", fontsize=12)

    plt.tight_layout()
    fig_path = output_dir / "categorical_vs_target.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()


def _plot_bivariate_top_correlated(df: pd.DataFrame, output_dir: Path) -> None:
    if "respuesta" not in df.columns:
        return

    output_dir = _ensure_dir(output_dir / "bivariate")

    corr = df.corr(numeric_only=True)
    if "respuesta" not in corr.columns:
        return

    corr_respuesta = corr["respuesta"].drop(labels=["respuesta"]).sort_values(key=lambda s: s.abs(), ascending=False)
    top_vars = corr_respuesta.head(6).index.tolist()

    n = len(top_vars)
    n_rows = 2
    n_cols = int(np.ceil(n / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    fig.suptitle(
        "Distribuciones de Variables Clave por Respuesta a Campaña",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    for idx, var in enumerate(top_vars):
        ax = axes[idx // n_cols, idx % n_cols]

        data_no = df[df["respuesta"] == 0][var].dropna()
        data_si = df[df["respuesta"] == 1][var].dropna()

        parts = ax.violinplot([data_no, data_si], positions=[0, 1], showmeans=True, showmedians=True, widths=0.7)
        for pc, color in zip(parts["bodies"], ["lightcoral", "lightgreen"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.boxplot(
            [data_no, data_si],
            positions=[0, 1],
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="white", alpha=0.5),
            medianprops=dict(color="darkred", linewidth=2),
        )

        media_no = data_no.mean()
        media_si = data_si.mean()
        mediana_no = data_no.median()
        mediana_si = data_si.median()
        diff_pct = ((media_si - media_no) / media_no * 100) if media_no != 0 else 0

        ax.text(
            0,
            ax.get_ylim()[1] * 0.95,
            f"No: μ={media_no:.1f}\nMd={mediana_no:.1f}",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.3),
        )
        ax.text(
            1,
            ax.get_ylim()[1] * 0.95,
            f"Sí: μ={media_si:.1f}\nMd={mediana_si:.1f}",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
        )

        color_diff = "green" if diff_pct > 0 else "red"
        ax.text(
            0.5,
            ax.get_ylim()[1] * 0.85,
            f"Δ: {diff_pct:+.1f}%",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=color_diff, alpha=0.2),
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No responde", "Responde"])
        ax.set_ylabel(var.replace("_", " ").title(), fontweight="bold")
        ax.set_title(f"{var.replace('_', ' ').title()}\n(r={corr_respuesta[var]:.3f})", fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    for j in range(len(top_vars), n_rows * n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])

    plt.tight_layout()
    fig_path = output_dir / "bivariate_violinplots.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()

    # Pruebas estadísticas resumidas en consola
    print("\n" + "=" * 80)
    print("ANÁLISIS ESTADÍSTICO BIVARIADO (Mann-Whitney U)")
    print("=" * 80)
    for var in top_vars:
        data_no = df[df["respuesta"] == 0][var].dropna()
        data_si = df[df["respuesta"] == 1][var].dropna()
        statistic, p_value = mannwhitneyu(data_no, data_si, alternative="two-sided")
        media_no = data_no.mean()
        media_si = data_si.mean()
        diff = media_si - media_no
        diff_pct = (diff / media_no * 100) if media_no != 0 else 0
        significancia = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(
            f"{var.upper():20s} | No={media_no:8.2f} Sí={media_si:8.2f} "
            f"Δ={diff:+8.2f} ({diff_pct:+5.1f}%) | p={p_value:0.4f} {significancia}"
        )


def run_additional_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Ejecuta el conjunto de gráficos avanzados sobre el dataset enriquecido.
    """
    print("=== 1.4 Análisis exploratorio visual avanzado ===")
    output_dir = _ensure_dir(output_dir)

    _plot_target_analysis(df, output_dir)
    _plot_correlation_heatmap(df, output_dir)
    _plot_outliers_boxplots(df, output_dir)
    _plot_categorical_vs_target(df, output_dir)
    _plot_bivariate_top_correlated(df, output_dir)


def main() -> None:
    """
    Punto de entrada opcional para ejecutar solo este módulo.
    """
    import main_eda  # type: ignore

    df_raw = main_eda.load_data()
    _, df_features = main_eda._apply_minimal_cleaning_and_feature_engineering(df_raw)  # type: ignore

    figures_dir = Path(__file__).resolve().parent / "figures" / "additional"
    run_additional_eda_plots(df_features, figures_dir)


if __name__ == "__main__":
    main()


