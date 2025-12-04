# Scripts Python de Análisis

## Descripción
Esta carpeta contiene todos los scripts Python modulares para el análisis del proyecto. Cada script tiene un propósito específico y puede ejecutarse independientemente, permitiendo un flujo de trabajo estructurado y reproducible.

## Filosofía de Diseño
- **Modularidad**: Cada script tiene una responsabilidad específica
- **Reproducibilidad**: Scripts que pueden ejecutarse en cualquier orden
- **Trazabilidad**: Guardado automático de datos en estados intermedios
- **Flexibilidad**: Fácil modificación y extensión

## Estructura de Scripts

### 01_EDA (Exploratory Data Analysis)
Scripts para análisis exploratorio de datos:
- `01_data_overview.py`: Vista general del dataset
- `02_descriptive_statistics.py`: Estadísticas descriptivas
- `03_missing_values_analysis.py`: Análisis de valores faltantes
- `04_outliers_detection.py`: Detección de outliers
- `05_correlation_analysis.py`: Análisis de correlaciones
- `06_data_visualization.py`: Visualizaciones exploratorias
- `07_feature_analysis.py`: Análisis detallado de features

### 02_Preprocessing
Scripts para preprocesamiento de datos:
- `01_data_cleaning.py`: Limpieza de datos
- `02_feature_engineering.py`: Creación de nuevas features
- `03_encoding_categorical.py`: Codificación de variables categóricas
- `04_scaling_normalization.py`: Escalado y normalización
- `05_train_test_split.py`: División de datos

### 03_Clustering
Scripts para modelos de clustering:
- `01_kmeans_analysis.py`: Análisis con K-Means
- `02_hierarchical_clustering.py`: Clustering jerárquico
- `03_dbscan_analysis.py`: Análisis con DBSCAN
- `04_clustering_evaluation.py`: Evaluación de clusters
- `05_cluster_interpretation.py`: Interpretación de resultados

### 04_Classification
Scripts para modelos de clasificación:
- `01_logistic_regression.py`: Regresión logística
- `02_random_forest.py`: Random Forest
- `03_svm_analysis.py`: Support Vector Machine
- `04_gradient_boosting.py`: Gradient Boosting
- `05_neural_networks.py`: Redes neuronales
- `06_classification_evaluation.py`: Evaluación de modelos

### 05_Regression
Scripts para modelos de regresión:
- `01_linear_regression.py`: Regresión lineal
- `02_polynomial_regression.py`: Regresión polinomial
- `03_ridge_lasso.py`: Ridge y Lasso
- `04_random_forest_regression.py`: Random Forest para regresión
- `05_xgboost_regression.py`: XGBoost para regresión
- `06_regression_evaluation.py`: Evaluación de modelos

### 06_Model_Comparison
Scripts para comparación de modelos:
- `01_performance_comparison.py`: Comparación de rendimiento
- `02_cross_validation.py`: Validación cruzada
- `03_hyperparameter_tuning.py`: Optimización de hiperparámetros
- `04_final_model_selection.py`: Selección de modelos finales

### 07_Results_and_Visualization
Scripts para resultados finales:
- `01_results_summary.py`: Resumen de resultados
- `02_business_insights.py`: Insights de negocio
- `03_final_visualizations.py`: Visualizaciones finales
- `04_model_interpretation.py`: Interpretación de modelos

## Gestión de Datos Intermedios

### Estrategia de Guardado
Cada script guarda automáticamente sus resultados en la carpeta `data/interim/` o `data/processed/` según corresponda:

```python
# Ejemplo de guardado en estado intermedio
import pandas as pd
import os

def save_intermediate_data(data, filename, description=""):
    """
    Guarda datos en estado intermedio con metadatos
    
    Args:
        data: DataFrame o objeto a guardar
        filename: Nombre del archivo
        description: Descripción del estado de los datos
    """
    interim_path = "data/interim/"
    os.makedirs(interim_path, exist_ok=True)
    
    # Guardar datos
    if isinstance(data, pd.DataFrame):
        data.to_csv(f"{interim_path}{filename}.csv", index=False)
    else:
        # Para otros objetos usar pickle
        import pickle
        with open(f"{interim_path}{filename}.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    # Guardar metadatos
    metadata = {
        'filename': filename,
        'description': description,
        'timestamp': pd.Timestamp.now(),
        'shape': data.shape if hasattr(data, 'shape') else None
    }
    
    with open(f"{interim_path}{filename}_metadata.json", 'w') as f:
        import json
        json.dump(metadata, f, default=str)
```

### Convenciones de Nomenclatura
- **Datos intermedios**: `{proceso}_{version}_{timestamp}.csv`
- **Metadatos**: `{filename}_metadata.json`
- **Modelos**: `{modelo}_{linea_trabajo}_{timestamp}.pkl`
- **Resultados**: `{modelo}_{metricas}_{timestamp}.json`

## Ejecución de Scripts

### Ejecución Individual
```bash
python scripts/01_EDA/01_data_overview.py
```

### Ejecución en Pipeline
```bash
python scripts/run_pipeline.py --stage eda
python scripts/run_pipeline.py --stage preprocessing
python scripts/run_pipeline.py --stage modeling
```

### Configuración
Todos los scripts utilizan configuraciones centralizadas en `config/config.yaml` para:
- Paths de datos
- Parámetros de modelos
- Configuraciones de evaluación
- Settings de logging

## Ventajas de este Enfoque
1. **Reproducibilidad**: Cada paso está documentado y puede reproducirse
2. **Modularidad**: Fácil modificación de componentes específicos
3. **Escalabilidad**: Fácil adición de nuevos análisis
4. **Colaboración**: Múltiples desarrolladores pueden trabajar en paralelo
5. **Debugging**: Fácil identificación y corrección de errores
6. **Versionado**: Control de versiones granular del código
