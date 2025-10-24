# Módulo de Datos

## Archivos
- `load_data.py`: Funciones para cargar y validar datos
- `preprocessing.py`: Pipeline de preprocesamiento
- `feature_engineering.py`: Creación y selección de features
- `validation.py`: Validación de integridad de datos

## Funciones Principales

### load_data.py
- `load_raw_data()`: Carga datos originales
- `validate_data()`: Valida estructura y tipos de datos
- `get_data_info()`: Información general del dataset

### preprocessing.py
- `clean_data()`: Limpieza básica de datos
- `handle_missing_values()`: Manejo de valores faltantes
- `detect_outliers()`: Detección de outliers
- `encode_categorical()`: Codificación de variables categóricas

### feature_engineering.py
- `create_features()`: Creación de nuevas features
- `select_features()`: Selección de features relevantes
- `scale_features()`: Escalado de variables numéricas

### validation.py
- `check_data_quality()`: Verificación de calidad de datos
- `validate_splits()`: Validación de divisiones train/test
