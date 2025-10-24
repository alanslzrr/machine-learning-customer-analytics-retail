"""
Ejemplo de Uso del Config.yaml
==============================

Este archivo muestra cómo usar las configuraciones del archivo config.yaml
en los scripts del proyecto.

IMPORTANTE: Este es solo un ejemplo de uso, no ejecutar directamente.
"""

# Ejemplo de importación y uso del config
# import yaml
# import os

def load_config_example():
    """
    Ejemplo de cómo cargar el archivo de configuración
    """
    # Cargar configuración
    # with open('config/config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
    
    # Ejemplos de uso:
    
    # 1. Obtener paths
    # data_path = config['paths']['data_raw']
    # models_path = config['paths']['models']
    
    # 2. Configuración de clustering
    # max_clusters = config['clustering']['kmeans']['max_clusters']
    # random_state = config['clustering']['kmeans']['random_state']
    
    # 3. Configuración de clasificación
    # target_var = config['classification']['target_variable']
    # test_size = config['classification']['test_size']
    
    # 4. Configuración de regresión
    # target_var = config['regression']['target_variable']
    # metrics = config['regression']['evaluation_metrics']
    
    # 5. Configuración de preprocesamiento
    # imputation_strategy = config['preprocessing']['imputation_strategy']
    # scaling_method = config['preprocessing']['scaling_method']
    
    pass

# Ejemplo de uso en un script:
def example_script_usage():
    """
    Ejemplo de cómo usar config en un script específico
    """
    # Cargar config
    # config = load_config()
    
    # Usar configuraciones específicas
    # random_state = config['general']['random_state']
    # test_size = config['classification']['test_size']
    # target_variable = config['classification']['target_variable']
    
    # Aplicar configuraciones
    # train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    pass
