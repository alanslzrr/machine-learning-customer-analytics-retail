Una pequeña documentacion para identificar los modelos:

Archivos a entregar (pipelines completos y metadatos):
- pipeline_clasificacion_sin_leakage.pkl   -> Pipeline final de clasificación (preprocesado + LogisticRegression con class_weight={0:1,1:7} y umbral que dertminamos 0.35).
- pipeline_regresion_sin_leakage.pkl       -> Pipeline final de regresión (preprocesado + GradientBoostingRegressor).
- pipeline_clustering.pkl                  -> Pipeline de clustering (escalado + KMeans K=2).
- pipeline_metadata.pkl                    -> Metadatos: columnas/features esperadas y configuración de los pipelines.