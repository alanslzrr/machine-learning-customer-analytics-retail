# Estructura Completa del Proyecto

## Resumen de la Estructura

```
aprendizaje_automatico/
├── README.md                          # Documentación principal del proyecto
├── requirements.txt                   # Dependencias de Python
├── .gitignore                        # Archivos a ignorar en Git
├── PROJECT_STRUCTURE.md              # Este archivo
│
├── data/                             # 📁 DATOS
│   ├── raw/                          # Datos originales
│   │   ├── proy_supermercado_dev.csv # Dataset principal
│   │   └── README.md
│   ├── processed/                    # Datos procesados
│   │   ├── clustering/               # Datos para clustering
│   │   ├── classification/           # Datos para clasificación
│   │   ├── regression/               # Datos para regresión
│   │   └── README.md
│   ├── interim/                      # Datos intermedios
│   │   └── README.md
│   └── README.md
│
├── scripts/                          # 📁 SCRIPTS PYTHON MODULARES
│   ├── 01_EDA/                       # Análisis Exploratorio
│   │   ├── 01_data_overview.py
│   │   ├── 02_descriptive_statistics.py
│   │   ├── 03_missing_values_analysis.py
│   │   ├── 04_outliers_detection.py
│   │   ├── 05_correlation_analysis.py
│   │   ├── 06_data_visualization.py
│   │   └── 07_feature_analysis.py
│   ├── 02_Preprocessing/             # Preprocesamiento
│   │   ├── 01_data_cleaning.py
│   │   ├── 02_feature_engineering.py
│   │   ├── 03_encoding_categorical.py
│   │   ├── 04_scaling_normalization.py
│   │   └── 05_train_test_split.py
│   ├── 03_Clustering/                # Modelos de Clustering
│   │   ├── 01_kmeans_analysis.py
│   │   ├── 02_hierarchical_clustering.py
│   │   ├── 03_dbscan_analysis.py
│   │   ├── 04_clustering_evaluation.py
│   │   └── 05_cluster_interpretation.py
│   ├── 04_Classification/            # Modelos de Clasificación
│   │   ├── 01_logistic_regression.py
│   │   ├── 02_random_forest.py
│   │   ├── 03_svm_analysis.py
│   │   ├── 04_gradient_boosting.py
│   │   ├── 05_neural_networks.py
│   │   └── 06_classification_evaluation.py
│   ├── 05_Regression/                # Modelos de Regresión
│   │   ├── 01_linear_regression.py
│   │   ├── 02_polynomial_regression.py
│   │   ├── 03_ridge_lasso.py
│   │   ├── 04_random_forest_regression.py
│   │   ├── 05_xgboost_regression.py
│   │   └── 06_regression_evaluation.py
│   ├── 06_Model_Comparison/          # Comparación de Modelos
│   │   ├── 01_performance_comparison.py
│   │   ├── 02_cross_validation.py
│   │   ├── 03_hyperparameter_tuning.py
│   │   └── 04_final_model_selection.py
│   ├── 07_Results_and_Visualization/ # Resultados y Visualizaciones
│   │   ├── 01_results_summary.py
│   │   ├── 02_business_insights.py
│   │   ├── 03_final_visualizations.py
│   │   └── 04_model_interpretation.py
│   ├── run_pipeline.py               # Pipeline principal de ejecución
│   └── README.md
│
├── src/                              # 📁 CÓDIGO FUENTE
│   ├── data/                         # Módulo de Datos
│   │   ├── load_data.py
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── validation.py
│   │   └── README.md
│   ├── models/                       # Módulo de Modelos
│   │   ├── clustering_models.py
│   │   ├── classification_models.py
│   │   ├── regression_models.py
│   │   ├── model_utils.py
│   │   └── README.md
│   ├── evaluation/                   # Módulo de Evaluación
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   ├── cross_validation.py
│   │   ├── hyperparameter_tuning.py
│   │   └── README.md
│   ├── utils/                        # Módulo de Utilidades
│   │   ├── config.py
│   │   ├── helpers.py
│   │   ├── logger.py
│   │   ├── file_utils.py
│   │   └── README.md
│   └── README.md
│
├── models/                           # 📁 MODELOS ENTRENADOS
│   ├── clustering/
│   │   ├── kmeans_model.pkl
│   │   ├── hierarchical_model.pkl
│   │   ├── dbscan_model.pkl
│   │   └── best_clustering_model.pkl
│   ├── classification/
│   │   ├── logistic_regression.pkl
│   │   ├── random_forest.pkl
│   │   ├── svm_model.pkl
│   │   ├── gradient_boosting.pkl
│   │   └── best_classification_model.pkl
│   ├── regression/
│   │   ├── linear_regression.pkl
│   │   ├── ridge_regression.pkl
│   │   ├── lasso_regression.pkl
│   │   ├── random_forest_regressor.pkl
│   │   ├── xgboost_regressor.pkl
│   │   └── best_regression_model.pkl
│   └── README.md
│
├── results/                          # 📁 RESULTADOS Y MÉTRICAS
│   ├── clustering/
│   │   ├── kmeans_results.csv
│   │   ├── silhouette_scores.json
│   │   ├── cluster_analysis.json
│   │   └── clustering_visualizations/
│   ├── classification/
│   │   ├── performance_metrics.csv
│   │   ├── confusion_matrices/
│   │   ├── roc_curves/
│   │   └── feature_importance.json
│   ├── regression/
│   │   ├── regression_metrics.csv
│   │   ├── residual_plots/
│   │   ├── prediction_plots/
│   │   └── coefficient_analysis.json
│   ├── comparison/
│   │   ├── model_comparison.csv
│   │   ├── cross_validation_results.json
│   │   └── hyperparameter_tuning_results.json
│   ├── visualizations/
│   │   ├── eda_plots/
│   │   ├── model_performance/
│   │   └── business_insights/
│   └── README.md
│
├── reports/                          # 📁 REPORTES Y DOCUMENTACIÓN
│   ├── final_report/
│   │   ├── technical_report.pdf
│   │   ├── executive_summary.md
│   │   ├── methodology.md
│   │   └── results_analysis.md
│   ├── presentations/
│   │   ├── final_presentation.pdf
│   │   ├── slides/
│   │   └── presentation_notes.md
│   ├── documentation/
│   │   ├── data_dictionary.md
│   │   ├── model_documentation.md
│   │   ├── code_documentation.md
│   │   └── experiment_log.md
│   ├── interim_reports/
│   │   ├── checkpoint_1_report.md
│   │   ├── checkpoint_2_report.md
│   │   └── progress_reports/
│   └── README.md
│
└── config/                           # 📁 CONFIGURACIONES
    ├── config.yaml
    ├── model_configs/
    │   ├── clustering_config.yaml
    │   ├── classification_config.yaml
    │   └── regression_config.yaml
    ├── experiment_configs/
    │   ├── experiment_1.yaml
    │   ├── experiment_2.yaml
    │   └── final_experiment.yaml
    ├── environment/
    │   ├── requirements.txt
    │   ├── environment.yml
    │   └── .env.example
    └── README.md
```

## Propósito de Cada Carpeta

### 📁 data/
Contiene todos los datos del proyecto organizados por estado de procesamiento:
- **raw/**: Datos originales sin modificar
- **processed/**: Datos limpios y preparados para modelado
- **interim/**: Datos en estados intermedios del procesamiento

### 📁 scripts/
Scripts Python modulares organizados por fase del proyecto:
- **01_EDA/**: Análisis exploratorio de datos
- **02_Preprocessing/**: Limpieza y preprocesamiento
- **03_Clustering/**: Modelos de clustering
- **04_Classification/**: Modelos de clasificación
- **05_Regression/**: Modelos de regresión
- **06_Model_Comparison/**: Comparación y selección de modelos
- **07_Results_and_Visualization/**: Resultados finales
- **run_pipeline.py**: Pipeline principal de ejecución

### 📁 src/
Código fuente modular y reutilizable:
- **data/**: Funciones para manejo de datos
- **models/**: Implementación de modelos de ML
- **evaluation/**: Métricas y evaluación
- **utils/**: Utilidades generales

### 📁 models/
Modelos entrenados listos para usar:
- Modelos de clustering, clasificación y regresión
- Mejores modelos de cada línea de trabajo
- Archivos pickle con pesos guardados

### 📁 results/
Resultados, métricas y visualizaciones:
- Métricas de rendimiento
- Gráficos y visualizaciones
- Comparaciones entre modelos

### 📁 reports/
Documentación y reportes:
- Informe técnico final
- Presentaciones
- Documentación del código

### 📁 config/
Configuraciones del proyecto:
- Archivos YAML con configuraciones
- Dependencias del entorno
- Configuraciones de experimentos

## Cumplimiento de Rúbricas

Esta estructura está diseñada para cumplir con todas las rúbricas del proyecto:

✅ **Análisis Exploratorio (EDA)** → `scripts/01_EDA/`
✅ **Preprocesamiento de Datos** → `scripts/02_Preprocessing/` + `src/data/`
✅ **Tres Líneas de Trabajo** → `scripts/03_Clustering/`, `04_Classification/`, `05_Regression/`
✅ **Evaluación de Modelos** → `scripts/06_Model_Comparison/` + `src/evaluation/`
✅ **Código Organizado** → `src/` con módulos bien estructurados
✅ **Modelos Entrenados** → `models/` con pesos guardados
✅ **Resultados Documentados** → `results/` + `reports/`
✅ **Reproducibilidad** → `config/` + `requirements.txt`
✅ **Pipeline Automatizado** → `scripts/run_pipeline.py`
