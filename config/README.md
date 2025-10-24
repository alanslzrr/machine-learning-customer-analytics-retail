# Configuraciones del Proyecto

## Descripción
Esta carpeta contiene archivos de configuración para el proyecto.

## Archivos

### config.yaml
Configuración principal del proyecto incluyendo:
- Paths de datos
- Parámetros de modelos
- Configuraciones de evaluación
- Configuraciones de logging

### model_configs/
- `clustering_config.yaml`: Configuraciones para modelos de clustering
- `classification_config.yaml`: Configuraciones para modelos de clasificación
- `regression_config.yaml`: Configuraciones para modelos de regresión

### experiment_configs/
- `experiment_1.yaml`: Configuración del primer experimento
- `experiment_2.yaml`: Configuración del segundo experimento
- `final_experiment.yaml`: Configuración del experimento final

### environment/
- `requirements.txt`: Dependencias del proyecto
- `environment.yml`: Entorno conda
- `.env.example`: Variables de entorno de ejemplo

## Uso
- Centralizar todas las configuraciones
- Facilitar la reproducibilidad
- Permitir cambios sin modificar código
- Versionar configuraciones de experimentos
