# Proyecto de Aprendizaje Automático - Supermercado

## Equipo de Trabajo
- **Enrique Posada Leiro**
- **Yésica Ramírez Bernal** 
- **Yago Ramos Sánchez**
- **Alan Ariel Salazar**

---

## Descripción del Proyecto

Este proyecto forma parte de la asignatura de Aprendizaje Automático de la UIE (Universidad Intercontinental de la Empresa). Trabajamos con un dataset real de una cadena de supermercados que busca optimizar su estrategia de marketing y personalización de la experiencia de cliente.

### Contexto del Negocio

La cadena de supermercados ha identificado la necesidad de mejorar su fidelización de clientes y personalizar la experiencia de compra. Para esto, necesitan:

1. **Identificar perfiles de clientes** mediante técnicas de clustering
2. **Predecir la propensión a responder a campañas** de marketing 
3. **Estimar el gasto anual** de los clientes para planificación estratégica

### Dataset

Trabajamos con `proy_supermercado_dev.csv`, que contiene información anonimizada de clientes del supermercado.

### Líneas de Trabajo (Según Especificaciones del Proyecto)

#### 1. Clustering
**Objetivo**: Identificar perfiles de clientes para personalización de experiencia
- Hacer emerger grupos de clientes con características similares
- Clusters deben ser fáciles de explicar e interpretar
- Variables: demográficas y de comportamiento

#### 2. Clasificación  
**Objetivo**: Clasificar clientes según propensión a responder a campañas de marketing
- Target: variable 'respuesta' (0/1)
- Identificar clientes con mayor probabilidad de respuesta positiva
- Optimizar eficiencia de estrategias de marketing

#### 3. Regresión
**Objetivo**: Predecir gasto anual de clientes
- Target: suma de gastos por categoría (gasto_vinos + gasto_frutas + gasto_carnes + gasto_pescado + gasto_dulces + gasto_oro)
- Información crucial para planificación estratégica

### Metodología

Nuestro enfoque sigue las mejores prácticas de Machine Learning:

1. **Análisis Exploratorio de Datos (EDA)**: Comprensión profunda del dataset
2. **Preprocesamiento**: Limpieza y transformación de datos
3. **Feature Engineering**: Creación de variables relevantes
4. **Modelado**: Implementación y evaluación de diferentes algoritmos
5. **Validación**: Evaluación robusta con técnicas de cross-validation
6. **Optimización**: Tuning de hiperparámetros

### Estructura del Proyecto

```
├── data/                    # Datos originales y procesados
├── scripts/                 # Scripts Python modulares de análisis
├── src/                     # Código fuente organizado
├── models/                  # Modelos entrenados
├── results/                 # Resultados y métricas
├── reports/                 # Informes y documentación
└── config/                  # Configuraciones del proyecto
```

### Herramientas y Tecnologías

- **Python** como lenguaje principal
- **Pandas, NumPy** para manipulación de datos
- **Scikit-learn** para algoritmos de ML
- **Matplotlib, Seaborn** para visualización
- **Scripts Python modulares** para análisis estructurado y reproducible

### Entregables

1. **Informe técnico** (máximo 10 páginas)
2. **Presentación final** (20 minutos)
3. **Código fuente** completo y reproducible
4. **Modelos entrenados** con pesos guardados

---

*Este proyecto representa nuestro trabajo colaborativo en el aprendizaje de técnicas de Machine Learning aplicadas a un caso de negocio real.*
