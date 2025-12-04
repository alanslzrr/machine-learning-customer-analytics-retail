# Modelos Entrenados

## Descripción
Esta carpeta contiene los modelos entrenados y sus pesos.

## Estructura

### clustering/
- `kmeans_model.pkl`: Modelo K-Means entrenado
- `hierarchical_model.pkl`: Modelo de clustering jerárquico
- `dbscan_model.pkl`: Modelo DBSCAN
- `best_clustering_model.pkl`: Mejor modelo de clustering

### classification/
- `logistic_regression.pkl`: Modelo de regresión logística
- `random_forest.pkl`: Modelo Random Forest
- `svm_model.pkl`: Modelo SVM
- `gradient_boosting.pkl`: Modelo Gradient Boosting
- `best_classification_model.pkl`: Mejor modelo de clasificación

### regression/
- `linear_regression.pkl`: Modelo de regresión lineal
- `ridge_regression.pkl`: Modelo Ridge
- `lasso_regression.pkl`: Modelo Lasso
- `random_forest_regressor.pkl`: Random Forest para regresión
- `xgboost_regressor.pkl`: Modelo XGBoost
- `best_regression_model.pkl`: Mejor modelo de regresión

## Formato de Archivos
- Los modelos se guardan en formato pickle (.pkl)
- Incluyen metadatos de entrenamiento
- Versiones con timestamps para trazabilidad

## Importante
- Estos archivos son necesarios para la evaluación final
- El profesor usará estos modelos para evaluar en datos no vistos
- Mantener versiones de los mejores modelos
