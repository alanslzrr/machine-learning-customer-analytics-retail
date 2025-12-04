"""
Configuración centralizada para el EDA del proyecto supermercado.

Este módulo:
- Carga el archivo `config/config.yaml`.
- Expone rutas tipadas a los datos crudos, intermedios y de salida.
- Define la carpeta base donde se guardarán las figuras del EDA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


# Directorios base
EDA_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = EDA_DIR.parent.parent  # .../aprendizaje_automatico
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> Dict[str, Any]:
    """
    Carga el archivo YAML de configuración del proyecto.

    Returns
    -------
    dict
        Configuración completa parseada desde `config/config.yaml`.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    return config


# Cargar configuración global una única vez
CONFIG: Dict[str, Any] = load_config()

PATHS: Dict[str, str] = CONFIG.get("paths", {})
EDA_SETTINGS: Dict[str, Any] = CONFIG.get("eda", {})


def _resolve_path(relative: str) -> Path:
    """
    Convierte una ruta relativa (desde config.yaml) a ruta absoluta
    respecto a la raíz del proyecto.
    """
    return (PROJECT_ROOT / relative).resolve()


# Directorios de datos
DATA_RAW_DIR: Path = _resolve_path(PATHS.get("data_raw", "data/raw/"))
DATA_INTERIM_DIR: Path = _resolve_path(PATHS.get("data_interim", "data/interim/"))
DATA_PROCESSED_DIR: Path = _resolve_path(PATHS.get("data_processed", "data/processed/"))

# Ficheros de datos utilizados en el notebook original
RAW_DATA_FILE: Path = DATA_RAW_DIR / "proy_supermercado_dev.csv"

# Fallback: si no existe en data/raw, intentar en la raíz del proyecto
if not RAW_DATA_FILE.exists():
    alt = PROJECT_ROOT / "proy_supermercado_dev.csv"
    if alt.exists():
        RAW_DATA_FILE = alt

CLEAN_DATA_FILE: Path = DATA_INTERIM_DIR / "supermercado_limpio.csv"
FEATURES_DATA_FILE: Path = DATA_INTERIM_DIR / "supermercado_features.csv"


# Carpeta base para figuras de EDA dentro de `scripts/01_EDA`
FIGURES_DIR: Path = EDA_DIR / "figures"


def ensure_eda_directories() -> None:
    """
    Crea, si no existen, los directorios necesarios para el EDA.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "CONFIG",
    "EDA_SETTINGS",
    "DATA_RAW_DIR",
    "DATA_INTERIM_DIR",
    "DATA_PROCESSED_DIR",
    "RAW_DATA_FILE",
    "CLEAN_DATA_FILE",
    "FEATURES_DATA_FILE",
    "FIGURES_DIR",
    "ensure_eda_directories",
]


