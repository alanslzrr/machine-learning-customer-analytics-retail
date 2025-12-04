"""
Orquestador del pipeline de EDA basado en el notebook `proyecto_00_eda[1].ipynb`.

Flujo a alto nivel:
1. Carga de datos crudos desde `proy_supermercado_dev.csv`.
2. Diagnóstico estructural inicial (overview).
3. Limpieza mínima e imputación (dataset limpio).
4. Ingeniería de características (dataset enriquecido).
5. EDA descriptivo, análisis de missing y visualizaciones avanzadas.

Todos los gráficos se guardan en subcarpetas de `scripts/01_EDA/figures/`.
Los datasets intermedios se guardan en `data/interim/`.
"""

from __future__ import annotations

from typing import Tuple
from pathlib import Path
import importlib.util

import pandas as pd


# Rutas base sin depender de que `scripts/01_EDA` sea un paquete importable
EDA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EDA_DIR.parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DATA_FILE = DATA_RAW_DIR / "proy_supermercado_dev.csv"
if not RAW_DATA_FILE.exists():
    alt = PROJECT_ROOT / "proy_supermercado_dev.csv"
    if alt.exists():
        RAW_DATA_FILE = alt

CLEAN_DATA_FILE = DATA_INTERIM_DIR / "supermercado_limpio.csv"
FEATURES_DATA_FILE = DATA_INTERIM_DIR / "supermercado_features.csv"
FIGURES_DIR = EDA_DIR / "figures"


def ensure_eda_directories() -> None:
    """
    Crea, si no existen, los directorios necesarios para el EDA.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """
    Carga el dataset original del supermercado.

    La lógica replica el comportamiento del notebook:
    - Usa `proy_supermercado_dev.csv` como fuente principal.
    - Permite que el fichero esté en `data/raw/` o en la raíz del proyecto.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos originales sin transformar.
    """
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset original en {RAW_DATA_FILE}. "
            "Verifica que el fichero `proy_supermercado_dev.csv` exista."
        )

    df = pd.read_csv(RAW_DATA_FILE)
    return df


def _apply_minimal_cleaning_and_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica las transformaciones de los pasos 1.2 y 1.3 del notebook:
    - Eliminación de identificadores sin valor predictivo.
    - Conversión de tipos.
    - Imputación de valores faltantes.
    - Eliminación de columnas constantes post-imputación.
    - Ingeniería de características demográficas, de gasto, comportamiento y hogar.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original cargado desde CSV.

    Returns
    -------
    (df_limpio, df_features)
        df_limpio: dataset con variables originales útiles tras limpieza.
        df_features: dataset enriquecido con todas las features derivadas.
    """
    df = df.copy()

    # 1.2.1 Eliminación de identificadores sin valor predictivo
    id_cols = [
        "id",
        "nombre",
        "apellidos",
        "direccion",
        "telefono1",
        "telefono2",
        "email",
        "dni",
        "tarjeta_credito_asociada",
    ]
    existing_id_cols = [c for c in id_cols if c in df.columns]
    if existing_id_cols:
        df = df.drop(columns=existing_id_cols)

    # 1.2.2 Conversión de tipos
    import numpy as np

    int_cols = [
        "anio_nacimiento",
        "hijos_casa",
        "adolescentes_casa",
        "recencia",
        "num_compras_oferta",
        "num_compras_web",
        "num_compras_catalogo",
        "num_compras_tienda",
        "num_visitas_web_mes",
        "reclama",
        "respuesta",
    ]
    existing_int_cols = [c for c in int_cols if c in df.columns]
    if existing_int_cols:
        df[existing_int_cols] = df[existing_int_cols].astype("Int64")

    gasto_cols = [c for c in df.columns if c.startswith("gasto_")]
    if gasto_cols:
        df[gasto_cols] = df[gasto_cols].astype("Int64")

    if "fecha_cliente" in df.columns:
        df["fecha_cliente"] = pd.to_datetime(df["fecha_cliente"], format="%d-%m-%Y", errors="coerce")

    bin_cols = ["acepta_cmp1", "acepta_cmp2", "acepta_cmp3", "acepta_cmp4", "acepta_cmp5"]
    for col in bin_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].map(
            {
                "1": 1,
                "1.0": 1,
                "SI": 1,
                "TRUE": 1,
                "0": 0,
                "0.0": 0,
                "NO": 0,
                "FALSE": 0,
                "NAN": None,
                "NONE": None,
                "<NA>": None,
                "": None,
            }
        )
        df[col] = df[col].astype("Int8")

    # 1.2.4 Imputación de valores faltantes
    num_cont_cols = df.select_dtypes(include=["float64"]).columns
    for col in num_cont_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    num_disc_cols = df.select_dtypes(include=["Int64", "Int8"]).columns
    for col in num_disc_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(int(median_val))

    if "fecha_cliente" in df.columns and df["fecha_cliente"].isna().sum() > 0:
        median_ts = df["fecha_cliente"].dropna().astype(np.int64).median()
        df["fecha_cliente"] = df["fecha_cliente"].fillna(pd.to_datetime(median_ts))

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    # 1.2.5 Eliminación de columnas constantes post-imputación
    const_cols_initial = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    if const_cols_initial:
        df = df.drop(columns=const_cols_initial)

    # En este punto consideramos df como "limpio"
    df_limpio = df.copy()

    # 1.3 Ingeniería de características
    # 1.3.1 Variables demográficas derivadas
    if "anio_nacimiento" in df.columns:
        ANIO_ACTUAL = 2025
        df["edad"] = ANIO_ACTUAL - df["anio_nacimiento"]

    if "fecha_cliente" in df.columns:
        FECHA_REFERENCIA = pd.Timestamp("2025-10-27")
        df["antiguedad_dias"] = (FECHA_REFERENCIA - df["fecha_cliente"]).dt.days
        df["antiguedad_anios"] = (df["antiguedad_dias"] / 365.25).round(1)

    # 1.3.2 Variables de gasto agregadas
    gasto_cols = [c for c in df.columns if c.startswith("gasto_")]
    if gasto_cols:
        df["gasto_total"] = df[gasto_cols].sum(axis=1)
        df["gasto_promedio"] = df[gasto_cols].mean(axis=1).round(2)
        for col in gasto_cols:
            nueva_col = col.replace("gasto_", "prop_gasto_")
            df[nueva_col] = (df[col] / df["gasto_total"]).fillna(0).round(3)
        df["categorias_compradas"] = (df[gasto_cols] > 0).sum(axis=1)

    # 1.3.3 Variables de comportamiento de compra
    required_cols = {"num_compras_web", "num_compras_catalogo", "num_compras_tienda", "num_compras_oferta"}
    if required_cols.issubset(df.columns):
        df["compras_totales"] = (
            df["num_compras_web"]
            + df["num_compras_catalogo"]
            + df["num_compras_tienda"]
            + df["num_compras_oferta"]
        )
        df["compras_online"] = df["num_compras_web"] + df["num_compras_catalogo"]
        df["compras_offline"] = df["num_compras_tienda"]

        df["tasa_compra_online"] = 0.0
        mask_compras = df["compras_totales"] > 0
        df.loc[mask_compras, "tasa_compra_online"] = (
            df.loc[mask_compras, "compras_online"] / df.loc[mask_compras, "compras_totales"]
        ).round(3)

        df["tasa_compra_oferta"] = 0.0
        df.loc[mask_compras, "tasa_compra_oferta"] = (
            df.loc[mask_compras, "num_compras_oferta"] / df.loc[mask_compras, "compras_totales"]
        ).round(3)

        df["ticket_promedio"] = 0.0
        df.loc[mask_compras, "ticket_promedio"] = (
            df.loc[mask_compras, "gasto_total"] / df.loc[mask_compras, "compras_totales"]
        ).round(2)

    # 1.3.4 Variables de composición del hogar
    if {"hijos_casa", "adolescentes_casa"}.issubset(df.columns):
        df["tamano_hogar"] = 1 + df["hijos_casa"] + df["adolescentes_casa"]
        df["total_dependientes"] = df["hijos_casa"] + df["adolescentes_casa"]
        df["tiene_dependientes"] = (df["total_dependientes"] > 0).astype(int)
        df["hogar_unipersonal"] = (df["tamano_hogar"] == 1).astype(int)

    # 1.3.5 Normalización de variables categóricas
    if "educacion" in df.columns:
        mapa_educacion = {
            "Basic": "Basica",
            "2n Cycle": "Secundaria",
            "Graduation": "Universitaria",
            "Master": "Master",
            "PhD": "Doctorado",
        }
        df["educacion"] = df["educacion"].replace(mapa_educacion)

    if "estado_civil" in df.columns:
        mapa_estado_civil = {
            "Married": "Casado",
            "Together": "Union_Libre",
            "Single": "Soltero",
            "Divorced": "Divorciado",
            "Widow": "Viudo",
            "Absurd": "Soltero",
            "YOLO": "Soltero",
            "Alone": "Soltero",
        }
        df["estado_civil"] = df["estado_civil"].replace(mapa_estado_civil)

    df_features = df.copy()
    return df_limpio, df_features


def run_eda_pipeline() -> None:
    """
    Ejecuta de principio a fin el pipeline de EDA:
    - Carga de datos crudos.
    - Overview estructural.
    - Limpieza + ingeniería de características.
    - Guardado de datasets intermedios.
    - (En módulos separados) estadísticas descriptivas, análisis de missing y gráficos.
    """
    print("=== EDA: inicializando ===")
    ensure_eda_directories()
    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo datos crudos desde: {RAW_DATA_FILE}")
    df_raw = load_data()
    print(f"Dataset original cargado con shape={df_raw.shape}")

    # Paso 1: overview básico (sin modificar df_raw)
    overview_fig_dir = FIGURES_DIR / "overview"
    overview_fig_dir.mkdir(parents=True, exist_ok=True)
    missing_fig_dir = FIGURES_DIR / "missing"
    missing_fig_dir.mkdir(parents=True, exist_ok=True)

    # Cargar módulos por ruta de archivo (nombres de fichero empiezan por dígito)
    def _load_local_module(filename: str, module_name: str):
        file_path = EDA_DIR / filename
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"No se pudo cargar el módulo {module_name} desde {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    data_overview = _load_local_module("01_data_overview.py", "eda_data_overview")
    missing_analysis = _load_local_module("03_missing_values_analysis.py", "eda_missing_values")

    # Overview y análisis de missing sobre datos crudos
    data_overview.run_data_overview(df_raw, overview_fig_dir)  # type: ignore[attr-defined]
    missing_analysis.run_missing_values_analysis(df_raw, missing_fig_dir)  # type: ignore[attr-defined]

    # Paso 2: limpieza + ingeniería de características
    print("Aplicando limpieza mínima e ingeniería de características...")
    df_limpio, df_features = _apply_minimal_cleaning_and_feature_engineering(df_raw)

    # Guardar datasets intermedios como en el notebook
    print(f"Guardando dataset limpio en: {CLEAN_DATA_FILE}")
    df_limpio.to_csv(CLEAN_DATA_FILE, index=False)

    print(f"Guardando dataset enriquecido en: {FEATURES_DATA_FILE}")
    df_features.to_csv(FEATURES_DATA_FILE, index=False)

    print("=== EDA: etapa estructural completada ===")
    print(f"  Dataset limpio:     {df_limpio.shape}")
    print(f"  Dataset enriquecido:{df_features.shape}")

    # Paso 3: estadísticas descriptivas y visualizaciones avanzadas
    descriptive_fig_dir = FIGURES_DIR / "descriptive"
    descriptive_fig_dir.mkdir(parents=True, exist_ok=True)
    additional_fig_dir = FIGURES_DIR / "additional"
    additional_fig_dir.mkdir(parents=True, exist_ok=True)

    descriptive_stats = _load_local_module("02_descriptive_statistics.py", "eda_descriptive_stats")
    additional_plots = _load_local_module("04_additional_eda_plots.py", "eda_additional_plots")

    descriptive_stats.run_descriptive_statistics(df_features, descriptive_fig_dir)  # type: ignore[attr-defined]
    additional_plots.run_additional_eda_plots(df_features, additional_fig_dir)  # type: ignore[attr-defined]


if __name__ == "__main__":
    run_eda_pipeline()


