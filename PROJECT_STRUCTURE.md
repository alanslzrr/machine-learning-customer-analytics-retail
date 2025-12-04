# Estructura del Proyecto

## Resumen de la Estructura

```
aprendizaje_automatico/
├── README.md                          # Documentación principal del proyecto
├── requirements.txt                   # Dependencias de Python
├── .gitignore                         # Archivos a ignorar en Git
├── PROJECT_STRUCTURE.md               # Este archivo
├── DOCUMENTACION_FINAL.md             # Informe técnico completo
├── PLAN_ACCION_CORRECCIONES.md        # Plan de mejoras metodológicas
│
├── proyecto_00_eda[1].ipynb           # 📓 Análisis Exploratorio de Datos
├── proyecto_01_preprocesamiento.ipynb # 📓 Limpieza y Feature Engineering
├── proyecto_02_modelos.ipynb          # 📓 Clustering, Clasificación y Regresión
│
├── data/                              # 📁 DATOS
│   ├── raw/                           # Datos originales
│   │   ├── proy_supermercado_dev.csv  # Dataset principal (1,989 clientes)
│   │   └── README.md
│   ├── interim/                       # Datos intermedios
│   │   ├── supermercado_limpio.csv    # Dataset tras limpieza
│   │   ├── supermercado_features.csv  # Dataset con features derivadas
│   │   └── README.md
│   ├── processed/                     # Datos finales procesados
│   │   ├── supermercado_preprocesado.csv
│   │   ├── supermercado_con_clusters.csv
│   │   ├── perfiles_clusters.csv
│   │   └── README.md
│   └── README.md
│
├── scripts/                           # 📁 SCRIPTS PYTHON
│   ├── 01_EDA/                        # Scripts de EDA funcionales
│   │   ├── main_eda.py                # Pipeline principal de EDA
│   │   ├── 00_eda_config.py           # Configuración
│   │   ├── 01_data_overview.py        # Vista general del dataset
│   │   ├── 02_descriptive_statistics.py
│   │   ├── 03_missing_values_analysis.py
│   │   ├── 04_additional_eda_plots.py # Visualizaciones adicionales
│   │   └── README.md
│   └── README.md
│
├── models/                            # 📁 MODELOS ENTRENADOS
│   ├── kmeans_model.pkl               # Modelo K-Means (4 clusters)
│   ├── gradient_boosting_regressor.pkl
│   └── README.md
│
├── results/                           # 📁 RESULTADOS Y MÉTRICAS
│   ├── comparacion_modelos_regresion.csv
│   ├── importancia_variables_regresion.csv
│   └── README.md
│
├── reports/                           # 📁 REPORTES
│   └── README.md
│
└── config/                            # 📁 CONFIGURACIONES
    ├── config.yaml                    # Configuración principal
    ├── config_usage_example.py        # Ejemplo de uso
    └── README.md
```

## Flujo de Trabajo

El proyecto sigue un flujo secuencial implementado en los notebooks:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. EDA                    2. PREPROCESAMIENTO      3. MODELOS      │
│  ─────────────────────     ─────────────────────    ─────────────── │
│                                                                     │
│  proyecto_00_eda[1]   →    proyecto_01_preproc  →   proyecto_02     │
│                                                                     │
│  • Vista general           • Limpieza               • Clustering    │
│  • Estadísticas            • Imputación             • Clasificación │
│  • Missing values          • Feature Engineering    • Regresión     │
│  • Correlaciones           • Encoding               • Evaluación    │
│  • Visualizaciones         • Escalado                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Descripción de Carpetas

### 📁 data/
Datos organizados por estado de procesamiento:
- **raw/**: Dataset original `proy_supermercado_dev.csv` (38 variables)
- **interim/**: Datos intermedios tras limpieza y feature engineering
- **processed/**: Datos finales listos para modelado (49 variables)

### 📁 scripts/01_EDA/
Scripts Python funcionales que replican el análisis del notebook EDA:
- `main_eda.py`: Orquestador del pipeline completo
- Scripts modulares para cada fase del análisis

### 📁 models/
Modelos entrenados serializados con pickle:
- K-Means para segmentación de clientes
- Gradient Boosting para regresión de gasto

### 📁 results/
Métricas y resultados de los modelos en formato CSV/JSON.

### 📁 config/
Configuración centralizada en formato YAML.

## Tres Líneas de Trabajo

| Línea | Objetivo | Algoritmo Final | Métrica Principal |
|-------|----------|-----------------|-------------------|
| **Clustering** | Segmentar clientes | K-Means (K=4) | Silhouette: 0.34 |
| **Clasificación** | Predecir respuesta a campañas | Gradient Boosting | AUC: 0.89 |
| **Regresión** | Predecir gasto anual | Gradient Boosting | R²: 0.85* |

*R² sin variables de pseudo-leakage (ticket_promedio, compras_totales)

## Ejecución

Los notebooks se ejecutan en orden:

```bash
# 1. Análisis Exploratorio
jupyter notebook proyecto_00_eda[1].ipynb

# 2. Preprocesamiento
jupyter notebook proyecto_01_preprocesamiento.ipynb

# 3. Modelado
jupyter notebook proyecto_02_modelos.ipynb
```

Alternativamente, el pipeline de EDA puede ejecutarse desde scripts:

```bash
cd scripts/01_EDA
python main_eda.py
```
