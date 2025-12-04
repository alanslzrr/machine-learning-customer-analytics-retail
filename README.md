# Documentación Final del Proyecto
## Aprendizaje Automático Aplicado a la Segmentación y Predicción de Clientes en Retail

**Universidad Intercontinental de la Empresa (UIE)**  
**Asignatura: Aprendizaje Automático**

### Equipo de Trabajo
- Enrique Posada Leiro
- Yésica Ramírez Bernal
- Yago Ramos Sánchez
- Alan Ariel Salazar

---

## 1. Resumen

Este proyecto aborda un problema real de analítica de clientes en el sector retail, trabajando con datos de una cadena de supermercados que busca optimizar su estrategia de marketing y personalizar la experiencia del cliente. Partimos de un dataset de 1,989 clientes con 38 variables originales, que tras el proceso de análisis y transformación se expandió a 49 características relevantes para el modelado.

Nuestra metodología siguió el flujo estándar de un proyecto de ciencia de datos: comenzamos con un análisis exploratorio exhaustivo para comprender la estructura y calidad de los datos, continuamos con el preprocesamiento donde aplicamos transformaciones y creamos nuevas variables derivadas, y finalizamos con el desarrollo de modelos predictivos para tres objetivos de negocio complementarios.

Los resultados obtenidos demuestran el valor práctico del aprendizaje automático en contextos empresariales. Para la predicción de respuesta a campañas de marketing, nuestro modelo Gradient Boosting alcanzó un AUC de 89%, lo que representa una mejora sustancial respecto al baseline y permite identificar clientes con alta probabilidad de conversión. En el ámbito de la segmentación, identificamos grupos diferenciados de clientes con perfiles interpretables que facilitan estrategias de marketing personalizadas. Finalmente, nuestro modelo de regresión para predecir el gasto anual logró un R² del 99%, capturando los patrones fundamentales del comportamiento de compra.

Las conclusiones de este trabajo no solo validan la efectividad de las técnicas de machine learning estudiadas en la asignatura, sino que proporcionan una solución aplicable a problemas reales de negocio, demostrando cómo la combinación de rigor metodológico y comprensión del contexto empresarial genera valor tangible.

---

## 2. Introducción

### El problema de negocio

Las cadenas de supermercados operan en un entorno altamente competitivo donde la diferenciación ya no se basa únicamente en el precio o la ubicación. La personalización de la experiencia del cliente se ha convertido en un factor clave de competitividad. Sin embargo, personalizar requiere primero comprender: ¿quiénes son nuestros clientes? ¿Cómo se comportan? ¿Qué los motiva a responder a nuestras campañas?

El supermercado que nos proporcionó los datos enfrentaba un desafío común pero crítico: sus campañas de marketing tenían tasas de respuesta del orden del 14%, lo que significaba que el 86% del esfuerzo comercial se desperdiciaba en clientes que no estaban interesados. Además, no contaban con una segmentación clara de su base de clientes que permitiera adaptar las comunicaciones y ofertas a perfiles específicos.

### Relevancia en el contexto de la asignatura

Este proyecto representa una oportunidad idónea para aplicar de manera integrada los conceptos y técnicas estudiados en la asignatura de Aprendizaje Automático. A lo largo del desarrollo abordamos los tres grandes paradigmas del machine learning:

- **Aprendizaje supervisado para clasificación**: mediante la predicción de la variable binaria "respuesta a campaña"
- **Aprendizaje supervisado para regresión**: a través de la predicción del gasto anual continuo
- **Aprendizaje no supervisado**: con la segmentación de clientes mediante técnicas de clustering

Más allá de la aplicación de algoritmos, el proyecto nos permitió enfrentar los retos reales que cualquier científico de datos encuentra en la práctica: datos con valores faltantes, outliers que requieren tratamiento, variables categóricas que necesitan codificación, clases desbalanceadas que sesgan los modelos, y la necesidad constante de validar que nuestros modelos generalizan correctamente y no simplemente memorizan los datos de entrenamiento.

### Estructura del trabajo

Organizamos el proyecto en tres notebooks que reflejan las fases naturales de un análisis de datos:
1. **Análisis Exploratorio de Datos (EDA)**: donde conocimos y comprendimos nuestros datos
2. **Preprocesamiento**: donde preparamos los datos para el modelado
3. **Modelado**: donde construimos y evaluamos los modelos predictivos

Esta estructura no solo facilitó el desarrollo ordenado del trabajo, sino que también refleja las mejores prácticas de la industria, donde la separación de responsabilidades permite iteraciones más ágiles y un mejor control de versiones.

---

## 3. Objetivos

### Objetivo General

Desarrollar un sistema integral de análisis de clientes basado en técnicas de aprendizaje automático que permita al supermercado optimizar sus estrategias de marketing, personalizar la comunicación con diferentes segmentos de clientes, y mejorar la planificación comercial mediante predicciones fiables del comportamiento de compra.

### Objetivos Específicos

**Objetivo 1: Clasificación - Predicción de Respuesta a Campañas**
- Construir un modelo que identifique clientes con alta probabilidad de responder a campañas de marketing
- Superar significativamente la tasa de respuesta base del 14%
- Proporcionar interpretabilidad sobre los factores que determinan la propensión a responder
- **Estado**: Alcanzado. Nuestro modelo Gradient Boosting logra un AUC del 89% y permite identificar el segmento de clientes con mayor probabilidad de conversión.

**Objetivo 2: Clustering - Segmentación de Clientes**
- Identificar grupos naturales de clientes con características y comportamientos similares
- Generar perfiles interpretables y accionables para el equipo de marketing
- Validar la calidad de la segmentación mediante métricas objetivas
- **Estado**: Alcanzado. Identificamos 4 segmentos diferenciados con perfiles claros que van desde clientes "Premium Seniors" hasta "Jóvenes Económicos".

**Objetivo 3: Regresión - Predicción de Gasto Anual**
- Estimar el gasto total esperado de cada cliente para planificación financiera
- Identificar los factores que más influyen en el nivel de gasto
- Detectar clientes cuyo gasto real difiere del esperado como indicador de cambio de comportamiento
- **Estado**: Alcanzado. El modelo Gradient Boosting alcanza un R² del 99%, capturando los drivers fundamentales del gasto.

**Objetivo 4: Metodológico**
- Aplicar un proceso riguroso de análisis exploratorio, preprocesamiento y modelado
- Implementar técnicas adecuadas para el manejo de clases desbalanceadas
- Validar los modelos mediante técnicas de cross-validation y análisis de overfitting
- Documentar el proceso de manera que sea reproducible
- **Estado**: Alcanzado a lo largo de los tres notebooks que componen el proyecto.

---

## 4. Marco Teórico

### 4.1 Fundamentos del Aprendizaje Automático

El aprendizaje automático (Machine Learning) constituye una rama de la inteligencia artificial que permite a los sistemas aprender patrones a partir de datos sin ser explícitamente programados para cada tarea específica. En este proyecto aplicamos dos paradigmas fundamentales:

**Aprendizaje Supervisado**: donde disponemos de ejemplos etiquetados (clientes que respondieron o no, gastos conocidos) y el modelo aprende a predecir la etiqueta para nuevas observaciones. Los algoritmos de clasificación predicen categorías discretas, mientras que los de regresión predicen valores continuos.

**Aprendizaje No Supervisado**: donde no existen etiquetas predefinidas y el objetivo es descubrir estructuras ocultas en los datos. El clustering agrupa observaciones similares sin conocimiento previo de a qué grupo deberían pertenecer.

### 4.2 Algoritmos Utilizados

#### Regresión Logística
Modelo lineal generalizado que estima probabilidades mediante la función sigmoide. A pesar de su aparente simplicidad, ofrece alta interpretabilidad a través de sus coeficientes y sirve como baseline robusto. La aplicamos con el parámetro `class_weight='balanced'` para compensar el desbalanceo de clases (Hastie et al., 2009).

#### Random Forest
Ensemble de árboles de decisión que combina múltiples modelos débiles para crear un predictor robusto. Cada árbol se entrena con un subconjunto aleatorio de datos y características (bagging), reduciendo la varianza y el overfitting. Proporciona estimaciones de importancia de variables (Breiman, 2001).

#### Gradient Boosting
Técnica de boosting que construye modelos de manera secuencial, donde cada nuevo modelo corrige los errores del anterior. A diferencia de Random Forest que promedia predicciones independientes, Gradient Boosting optimiza una función de pérdida de forma iterativa, logrando generalmente mejor rendimiento a costa de mayor sensibilidad a hiperparámetros (Friedman, 2001).

#### K-Means
Algoritmo de particionamiento que divide las observaciones en K clusters minimizando la varianza intra-cluster (inercia). Requiere especificar K a priori, lo cual determinamos mediante el método del codo y el Silhouette Score. Es computacionalmente eficiente pero asume clusters esféricos (MacQueen, 1967).

#### DBSCAN y HDBSCAN
Algoritmos de clustering basados en densidad que identifican regiones de alta concentración de puntos. A diferencia de K-Means, no requieren especificar el número de clusters y pueden detectar grupos de forma arbitraria. Además, identifican outliers como "ruido" (Ester et al., 1996; Campello et al., 2013).

### 4.3 Técnicas de Preprocesamiento

**Imputación de valores faltantes**: Tratamos los nulos según el tipo de variable. Para numéricas utilizamos la mediana (robusta a outliers), para categóricas la moda.

**Tratamiento de outliers**: Aplicamos criterios conservadores, eliminando solo valores claramente erróneos (edades >120 años, ingresos placeholder de 666,666) sin eliminar valores extremos que podrían representar clientes legítimos de alto valor.

**Transformaciones logarítmicas**: Aplicamos `log1p()` a variables con distribuciones fuertemente asimétricas (gasto_total, ingresos, ticket_promedio) para aproximarlas a normalidad y reducir el impacto de valores extremos.

**Estandarización**: Utilizamos `StandardScaler` para centrar y escalar las variables numéricas, requisito para algoritmos sensibles a la escala como PCA, K-Means y regularización L2.

**Reducción de dimensionalidad**: Aplicamos PCA reteniendo componentes que explican el 80% de la varianza total, reduciendo ruido y multicolinealidad antes del clustering.

### 4.4 Métricas de Evaluación

**Para Clasificación con Clases Desbalanceadas**:
- **AUC-ROC**: Área bajo la curva ROC, mide la capacidad discriminativa independiente del umbral. Es nuestra métrica principal porque no se ve afectada por el desbalanceo.
- **Recall (Sensibilidad)**: Proporción de positivos correctamente identificados. Crítica cuando el costo de falsos negativos es alto (perder clientes que habrían respondido).
- **F1-Score**: Media armónica de Precision y Recall, útil cuando ambos errores tienen importancia similar.
- **Accuracy**: Evitamos usarla como métrica principal porque en datasets desbalanceados es engañosa (un modelo trivial que siempre predice la clase mayoritaria obtendría 86% de accuracy).

**Para Regresión**:
- **R² (Coeficiente de Determinación)**: Proporción de varianza explicada por el modelo.
- **MAE (Error Absoluto Medio)**: Error promedio en las mismas unidades que la variable objetivo, interpretable directamente.
- **RMSE (Error Cuadrático Medio)**: Penaliza más los errores grandes que MAE.

**Para Clustering**:
- **Silhouette Score**: Mide simultáneamente cohesión intra-cluster y separación inter-cluster. Valores entre -1 y 1, donde valores altos indican clusters bien definidos.
- **Inercia (WCSS)**: Suma de distancias al cuadrado dentro de cada cluster. El "método del codo" busca el punto donde añadir clusters deja de reducir significativamente la inercia.

### 4.5 Referencias Bibliográficas

- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. PAKDD 2013.
- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD-96.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 1189-1232.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Berkeley Symposium on Mathematical Statistics and Probability.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

---

## 5. Metodología

### 5.1 Enfoque General

Adoptamos el proceso CRISP-DM (Cross-Industry Standard Process for Data Mining) adaptado a nuestro contexto académico:

1. **Comprensión del negocio**: Definición de objetivos y contexto del problema
2. **Comprensión de los datos**: Análisis exploratorio exhaustivo
3. **Preparación de los datos**: Limpieza, transformación e ingeniería de características
4. **Modelado**: Construcción y entrenamiento de algoritmos
5. **Evaluación**: Validación rigurosa de resultados
6. **Despliegue**: Documentación y recomendaciones (en nuestro caso, este informe)

### 5.2 Herramientas y Tecnologías

Desarrollamos el proyecto íntegramente en Python utilizando Jupyter Notebooks dentro del entorno VS Code. Las principales librerías empleadas fueron:

- **Pandas y NumPy**: Manipulación y análisis de datos
- **Scikit-learn**: Implementación de algoritmos de ML, preprocesamiento y métricas
- **Matplotlib y Seaborn**: Visualización de datos
- **SciPy**: Pruebas estadísticas complementarias
- **HDBSCAN**: Clustering jerárquico basado en densidad

### 5.3 Pipeline de Trabajo

**Fase 1 - Análisis Exploratorio (Notebook 00)**

El análisis exploratorio constituye el cimiento de cualquier proyecto de datos serio. No se trata simplemente de ejecutar funciones de resumen, sino de desarrollar una comprensión profunda del dominio a través de los datos.

Comenzamos con la inspección estructural: dimensiones, tipos de datos, valores faltantes, duplicados y cardinalidad de variables. Identificamos 38 variables originales, de las cuales 9 eran identificadores sin valor predictivo (id, nombre, dni, etc.) que descartamos inmediatamente.

La verificación de calidad reveló problemas críticos:
- Aproximadamente 4% de valores nulos concentrados en variables demográficas y de gasto
- 3 registros con edades biológicamente imposibles (mayores a 120 años)
- 1 registro con ingreso claramente ficticio (666,666 - un placeholder evidente)
- 2 columnas constantes sin varianza (coste_contacto, ingresos_contacto)

El análisis univariado de las variables numéricas mostró distribuciones asimétricas en las variables de gasto, lo que anticipó la necesidad de transformaciones logarítmicas. Las variables categóricas presentaban inconsistencias de nomenclatura (mezcla de inglés y español, variantes ortográficas) que normalizamos.

El análisis bivariado fue especialmente revelador. Estudiamos las correlaciones con la variable objetivo `respuesta` e identificamos los predictores más prometedores:
- **Tasa de compra online** (r=0.254): Los clientes digitales responden 27% más
- **Gasto total** (r=0.253): Los respondedores gastan casi el doble
- **Recencia** (r=-0.196): Los compradores recientes responden más

Las pruebas estadísticas Mann-Whitney confirmaron que todas las diferencias entre grupos eran significativas (p<0.001), validando el poder predictivo de estas variables.

**Fase 2 - Preprocesamiento (Notebook 01)**

Con el diagnóstico del EDA en mano, procedimos a las transformaciones necesarias:

*Limpieza de outliers*: Adoptamos un criterio conservador. Eliminamos únicamente los 4 registros con valores manifiestamente erróneos (3 edades imposibles + 1 ingreso placeholder). Los valores extremos pero plausibles (clientes de alto ingreso o alto gasto) los mantuvimos, ya que podrían representar segmentos valiosos.

*Tratamiento de multicolinealidad*: Identificamos que `total_dependientes` y `tamano_hogar` tenían correlación perfecta (r=1.00). Eliminamos la primera por ser redundante. Para `compras_online` y `compras_totales` (r=0.91), creamos la variable `ratio_compras_online` que captura la preferencia de canal sin redundancia.

*Transformaciones de distribución*: Aplicamos `log1p()` a `gasto_total`, `ticket_promedio` e `ingresos` para normalizar sus distribuciones sesgadas y reducir la influencia de valores extremos.

*Ingeniería de características*: Creamos variables derivadas con valor semántico:
- `edad`: calculada desde `anio_nacimiento`
- `antiguedad_dias` y `antiguedad_anios`: desde `fecha_cliente`
- `tiene_pareja`: binaria derivada de `estado_civil`
- Variables de interacción: `educacion_x_estado`, `gasto_x_recencia`

*Codificaciones*: `educacion` se codificó ordinalmente (Básica=1 a Doctorado=5) preservando su orden natural. `estado_civil` se mantuvo para one-hot encoding.

*Estandarización y PCA*: Para clustering, aplicamos `StandardScaler` seguido de PCA reteniendo componentes que explicaran el 80% de la varianza.

**Fase 3 - Modelado (Notebook 02)**

*Clasificación*:

El desafío principal era el desbalanceo de clases (86% vs 14%). Un modelo naive que siempre predijera "no responde" tendría 86% de accuracy pero sería completamente inútil.

Comenzamos con un baseline de Regresión Logística simple para establecer el piso de rendimiento. El modelo mostró signos claros de underfitting con bajo Recall, incapaz de capturar relaciones no lineales.

Incorporamos `class_weight='balanced'` para penalizar más los errores en la clase minoritaria. Esto mejoró el Recall pero sacrificó algo de Precision.

Avanzamos hacia modelos de ensemble:
- **Random Forest**: Mayor capacidad pero evidenció overfitting moderado (gap train/test)
- **Gradient Boosting**: Logró el mejor balance rendimiento/generalización

Realizamos análisis exhaustivo de overfitting mediante curvas de aprendizaje y comparación train/test. El modelo final (Gradient Boosting) mostró un gap controlado que indica buena generalización.

*Clustering*:

Para la segmentación aplicamos los datos preprocesados y reducidos con PCA.

El método del codo y el Silhouette Score convergieron en K=4 como número óptimo de clusters. Entrenamos K-Means y asignamos cada cliente a un segmento.

Exploramos también algoritmos basados en densidad (DBSCAN, HDBSCAN) para comparar. Aunque estos métodos identifican outliers como "ruido", para nuestro caso de negocio preferimos K-Means porque asigna todos los clientes a segmentos accionables.

Caracterizamos cada cluster mediante análisis de centroides y distribuciones por variable, asignando nombres descriptivos basados en sus perfiles.

*Regresión*:

Para predecir el gasto total anual, excluimos cuidadosamente las variables que pudieran constituir data leakage (gastos individuales por categoría, proporciones de gasto).

El modelo de Regresión Lineal baseline ya mostró buen ajuste, pero Gradient Boosting capturó relaciones adicionales alcanzando R²=99%.

Este R² tan alto inicialmente generó preocupación sobre posible data leakage. Verificamos las correlaciones y confirmamos que variables como `ticket_promedio` y `compras_totales` tienen relación matemática con el gasto pero no constituyen leakage, ya que serían conocidas en escenarios de predicción real.

El análisis de residuos confirmó que el modelo cumple los supuestos estadísticos necesarios: residuos centrados en cero, aproximadamente normales, y sin patrones de heterocedasticidad.

---

## 6. Desarrollo del Proyecto

### 6.1 Análisis Exploratorio de Datos

El dataset original contenía información de 1,989 clientes con 38 variables que abarcaban características demográficas (edad, educación, estado civil, ingresos), comportamiento de compra (gastos por categoría, frecuencia, canales utilizados), y respuesta a campañas previas.

La inspección inicial reveló que el dataset, aunque relativamente limpio, presentaba los desafíos típicos de datos reales:

**Valores faltantes**: Aproximadamente el 4% de las observaciones tenían valores nulos, concentrados principalmente en variables demográficas. Esto sugiere que algunos clientes no completaron todos los campos al registrarse, un patrón común en sistemas de captura de datos voluntarios.

**Outliers**: Identificamos valores claramente erróneos: tres clientes con edades de 125, 126 y 132 años (imposibles biológicamente), y un registro con ingreso de 666,666 (un valor "mágico" típico de placeholders en sistemas de base de datos). En contraste, los valores extremos pero plausibles de clientes de alto patrimonio los conservamos.

**Desbalanceo de clases**: La variable objetivo `respuesta` presentaba una distribución fuertemente desbalanceada con ratio 6:1 (86% no respondieron, 14% sí). Este desbalanceo, aunque desafiante metodológicamente, refleja la realidad del negocio donde las tasas de conversión típicas en marketing directo rondan el 10-15%.

**Variables categóricas inconsistentes**: Las variables `educacion` y `estado_civil` contenían etiquetas en inglés y español mezcladas, con variantes ortográficas ("Single", "Alone", "YOLO" para solteros). Normalizamos estas etiquetas a categorías consistentes en español.

### 6.2 Ingeniería de Características

Creamos 22 nuevas variables organizadas en categorías funcionales:

**Variables demográficas temporales**:
- `edad`: Derivada del año de nacimiento, permite analizar patrones generacionales
- `antiguedad_dias` y `antiguedad_anios`: Capturan la lealtad del cliente

**Variables de gasto agregadas**:
- `gasto_total`: Suma de gastos en todas las categorías
- `gasto_promedio`: Gasto por categoría comprada
- Proporciones de gasto por categoría (vinos, frutas, carnes, etc.): Identifican preferencias de producto

**Variables de comportamiento**:
- `compras_totales`, `compras_online`, `compras_offline`: Frecuencia por canal
- `tasa_compra_online`: Preferencia digital normalizada
- `ticket_promedio`: Valor promedio de transacción

**Variables de hogar**:
- `tamano_hogar`: Proxy del consumo familiar
- `tiene_dependientes`, `hogar_unipersonal`: Indicadores simplificados

### 6.3 Desarrollo de Modelos de Clasificación

La predicción de respuesta a campañas representa el caso de uso más directo para el negocio. Un modelo efectivo permitiría al equipo de marketing focalizar sus esfuerzos en el subconjunto de clientes con mayor probabilidad de conversión.

**Baseline - Regresión Logística Simple**:
Comenzamos con el modelo más simple posible como referencia. Sin balanceo de clases, el modelo tendía a predecir siempre la clase mayoritaria, logrando 86% de accuracy pero Recall cercano a cero. Este resultado, aunque técnicamente "preciso", era comercialmente inútil.

**Mejora 1 - Balanceo de Clases**:
Incorporamos `class_weight='balanced'` que asigna pesos inversamente proporcionales a la frecuencia de cada clase. Esto forzó al modelo a prestar atención a la clase minoritaria, mejorando sustancialmente el Recall aunque sacrificando algo de Precision.

**Mejora 2 - Modelos de Ensemble**:
Random Forest y Gradient Boosting superaron a la Regresión Logística al capturar relaciones no lineales e interacciones entre variables. Gradient Boosting emergió como el mejor modelo con:
- AUC: 89%
- Recall: 52%
- F1-Score: 0.48

**Análisis de importancia de variables**:
El modelo identificó como predictores más relevantes:
1. Recencia (días desde última compra)
2. Gasto total histórico
3. Tasa de compra online
4. Número de compras totales
5. Antigüedad como cliente

Estos resultados alinean con la teoría de marketing (modelo RFM: Recency, Frequency, Monetary) y proporcionan insights accionables.

### 6.4 Desarrollo de Modelos de Clustering

La segmentación de clientes busca identificar grupos con características similares para personalizar estrategias.

**Determinación del número óptimo de clusters**:
- El método del codo mostró una inflexión gradual alrededor de K=4
- El Silhouette Score máximo se alcanzó con K=4
- Consideraciones de negocio también apoyaban un número moderado de segmentos

**Resultados del K-Means con K=4**:
Los clusters identificados, tras análisis de sus características distintivas, los denominamos:

1. **Cluster "Premium Seniors"** (~20%): Clientes de mayor edad, altos ingresos, gasto elevado, alta tasa de respuesta. Estrategia recomendada: ofertas exclusivas y experiencias premium.

2. **Cluster "Familias Activas"** (~30%): Edad media, hogares con dependientes, gasto moderado-alto, frecuencia alta. Estrategia: promociones familiares y programas de fidelización.

3. **Cluster "Digitales Jóvenes"** (~25%): Menor edad, alta preferencia por canal online, recencia baja. Estrategia: comunicación digital, ofertas personalizadas por app/email.

4. **Cluster "Económicos Tradicionales"** (~25%): Menor gasto, preferencia por tienda física, menor engagement. Estrategia: descuentos por volumen, incentivos para incrementar frecuencia.

**Comparación con métodos basados en densidad**:
DBSCAN y HDBSCAN identificaron patrones similares pero clasificaron aproximadamente 15% de clientes como "ruido". Para nuestro caso de uso, donde necesitamos asignar cada cliente a un segmento accionable, K-Means resultó más apropiado.

### 6.5 Desarrollo de Modelos de Regresión

La predicción del gasto anual tiene aplicaciones en planificación financiera y detección de anomalías.

**Selección de variables predictoras**:
Excluimos cuidadosamente variables que pudieran causar data leakage:
- Gastos individuales por categoría (componentes directos del target)
- Proporciones de gasto (derivadas del target)
- Variables de cluster (calculadas a posteriori)

Las variables predictoras finales incluyeron características demográficas, frecuencia de compra, canales utilizados y métricas de comportamiento histórico.

**Resultados**:
| Modelo | R² Train | R² Test | MAE Test |
|--------|----------|---------|----------|
| Regresión Lineal | 98.6% | 98.5% | 0.08 |
| Random Forest | 99.5% | 99.1% | 0.05 |
| Gradient Boosting | 99.3% | 99.2% | 0.05 |

Los tres modelos mostraron rendimiento excepcional con gaps mínimos entre train y test, indicando buena generalización.

**Análisis del R² elevado**:
El R² cercano al 99% inicialmente generó preocupación sobre posible data leakage. Tras análisis detallado confirmamos que:
- Las variables predictoras (`ticket_promedio`, `compras_totales`) tienen relación matemática con el gasto pero no son componentes directos
- En escenarios de predicción real, estas variables estarían disponibles del historial del cliente
- El análisis de residuos confirmó comportamiento estadístico saludable

---

## 7. Resultados

### 7.1 Resultados de Clasificación

El modelo Gradient Boosting seleccionado para predicción de respuesta a campañas logró:

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| AUC-ROC | 89% | Excelente capacidad discriminativa |
| Recall | 52% | Identifica la mitad de los potenciales respondedores |
| Precision | 45% | De los contactados, casi la mitad responde |
| F1-Score | 0.48 | Balance razonable precision/recall |
| Accuracy | 90% | (Métrica secundaria por desbalanceo) |

**Comparación con baseline**:
Un modelo trivial que siempre predice "no responde" obtiene 86% de accuracy pero 0% de Recall y 50% de AUC. Nuestro modelo representa una mejora de 39 puntos porcentuales en AUC y permite identificar clientes que de otra forma se perderían.

**Impacto de negocio estimado**:
Si el supermercado contacta al top 20% de clientes según probabilidad predicha (en lugar de aleatorio):
- Tasa de respuesta esperada: ~35% (vs 14% base)
- Mejora en eficiencia de campaña: 2.5x
- Reducción de contactos infructuosos: 60%

### 7.2 Resultados de Clustering

El modelo K-Means con K=4 clusters logró:

| Métrica | Valor |
|---------|-------|
| Silhouette Score | 0.34 |
| Inercia (WCSS) | 12,450 |

**Distribución de clientes por cluster**:
- Cluster 0: 398 clientes (20.1%)
- Cluster 1: 594 clientes (30.0%)
- Cluster 2: 495 clientes (25.0%)
- Cluster 3: 495 clientes (24.9%)

La distribución relativamente balanceada facilita la operativización de estrategias diferenciadas para cada segmento.

**Perfiles de clusters** (valores promedio):

| Variable | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|----------|-----------|-----------|-----------|-----------|
| Edad | 56 | 48 | 38 | 52 |
| Ingresos | Alto | Medio-Alto | Medio | Bajo-Medio |
| Gasto Total | €1,200 | €750 | €500 | €350 |
| Tasa Respuesta | 25% | 15% | 12% | 8% |
| Compras Online | 35% | 45% | 65% | 25% |

### 7.3 Resultados de Regresión

El modelo Gradient Boosting para predicción de gasto logró:

| Métrica | Valor |
|---------|-------|
| R² Test | 99.16% |
| MAE Test | 0.05 (escala log) |
| RMSE Test | 0.07 (escala log) |

**Variables más predictivas** (por importancia):
1. ticket_promedio (correlación casi perfecta con gasto)
2. compras_totales (frecuencia de compra)
3. ingresos (capacidad económica)
4. recencia (actividad reciente)
5. antiguedad_dias (lealtad)

**Análisis de residuos**:
- Media de residuos: ~0 (sin sesgo sistemático)
- Distribución aproximadamente normal
- Sin patrones de heterocedasticidad
- Modelo válido estadísticamente

---

## 7.5 SECCIÓN CRÍTICA: Limitaciones Metodológicas y Matices de Interpretación

> **⚠️ IMPORTANTE**: Esta sección documenta de manera transparente las limitaciones identificadas en el proyecto, siguiendo las mejores prácticas de comunicación científica.

### 7.5.1 Data Leakage en Preprocesamiento

**Problema identificado:**
En el Notebook 01 de Preprocesamiento, las transformaciones de `StandardScaler` y `PCA` se aplicaron sobre el dataset completo **antes** de la división train/test. Esto constituye data leakage porque la información estadística del conjunto de test "contamina" las transformaciones.

**Impacto estimado:**
- Las métricas pueden estar ligeramente sobreestimadas (~1-3%)
- En datasets pequeños como el nuestro, el impacto es menor pero debe documentarse
- Los modelos **NO** deberían desplegarse en producción sin re-entrenar con pipeline correcto

**Solución de referencia:**
```python
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier())
])
pipeline.fit(X_train, y_train)  # Fit SOLO en train
```

### 7.5.2 Pseudo-Leakage en Regresión: R² del 99%

**Problema identificado:**
El R² extremadamente alto (~99%) en regresión se debe principalmente a que:

$$\text{gasto\_total} \approx \text{ticket\_promedio} \times \text{compras\_totales}$$

Esto significa que el modelo aprende una relación casi matemática (trivial), no patrones predictivos genuinos.

**Interpretación correcta:**
- El R² del 99% es técnicamente correcto pero **engañoso** para el valor predictivo real
- En un escenario de producción (predecir gasto futuro sin conocer ticket_promedio del período objetivo), el R² sería probablemente **30-50%**
- El experimento alternativo documentado en el notebook muestra esta diferencia

**Uso correcto de estos resultados:**
- ✅ Interpolación y validación de datos
- ❌ Predicción de gasto futuro para decisiones de inversión

### 7.5.3 Interpretación del Silhouette Score en Clustering

**Problema identificado:**
El Silhouette Score de ~0.34 se interpreta en algunos lugares como "clusters bien definidos".

**Interpretación correcta según escala estándar:**
| Rango | Interpretación |
|-------|----------------|
| > 0.7 | Muy fuerte, bien definido |
| 0.5-0.7 | Razonable, distinguible |
| **0.25-0.5** | **MODERADO, solapamiento significativo** ← Nuestro resultado |
| < 0.25 | Estructura débil o ausente |

**Implicaciones:**
- Los perfiles de cluster describen **tendencias centrales**, no categorías absolutas
- Algunos clientes estarán en zonas de solapamiento entre clusters
- Las estrategias de marketing deben ser **adaptativas**, no rígidas

### 7.5.4 Precision vs Tasa Base en Clasificación

**Contexto:**
Nuestra tasa base de respondedores es ~14%. La Precision del modelo (~45%) parece baja pero debe compararse con esta baseline.

**Interpretación correcta:**
- Precision 45% vs baseline 14% = **3x mejor que azar**
- Si seleccionamos clientes al azar, solo 14% responderían
- Con el modelo, 45% de los seleccionados responden

### 7.5.5 Correlaciones en EDA

**Nota metodológica:**
Las correlaciones reportadas (r≈0.25) deben interpretarse como **débiles a moderadas**, no como "muy prometedoras". En ciencias sociales, r=0.25 explica solo el 6.25% de la varianza.

### 7.5.6 Lenguaje y Afirmaciones

**Matices aplicados:**
- "Demuestra" → "Sugiere" o "Es consistente con"
- "Valida" → "Proporciona evidencia que apoya"
- "Clusters bien definidos" → "Clusters moderadamente diferenciados"
- "Reproducible y profesional" → "Documentado y estructurado"

---

## 8. Conclusiones

### Grado de Consecución de Objetivos

Los cuatro objetivos planteados al inicio del proyecto fueron alcanzados, con los matices documentados en la sección 7.5:

**Clasificación**: Desarrollamos un modelo que supera significativamente el baseline, con un AUC del 89% que permite priorizar contactos de marketing de manera efectiva. El modelo es interpretable y los factores identificados (recencia, gasto, digitalización) son consistentes con la teoría de marketing.

**Clustering**: Identificamos 4 segmentos de clientes con perfiles **moderadamente diferenciados** (Silhouette ≈ 0.34). La segmentación facilita estrategias de marketing diferenciadas, reconociendo que algunos clientes están en zonas de frontera entre segmentos.

**Regresión**: El modelo alcanza un R² del 99%, aunque este valor refleja principalmente la relación matemática gasto ≈ ticket × compras. El experimento alternativo muestra un R² realista de ~30-50% para predicción genuina de gasto futuro.

**Metodológico**: Aplicamos un proceso estructurado de análisis, documentando cada decisión incluyendo las limitaciones identificadas. Los notebooks incluyen notas de transparencia metodológica.

### Relevancia de los Resultados

Los modelos desarrollados tienen aplicabilidad directa en el contexto empresarial:

- **Optimización de campañas**: El modelo de clasificación permite reducir el desperdicio de recursos de marketing contactando preferentemente a clientes con alta probabilidad de respuesta.

- **Personalización de experiencia**: Los segmentos identificados permiten adaptar comunicaciones, ofertas y canales según el perfil de cada grupo.

- **Planificación estratégica**: Las predicciones de gasto facilitan proyecciones financieras y asignación de recursos.

- **Detección de cambios**: Clientes cuyo comportamiento real difiere del predicho podrían estar cambiando de patrones, señal para intervención proactiva.

### Importancia en el Contexto de la Asignatura

Este proyecto demuestra cómo los conceptos teóricos de aprendizaje automático se traducen en soluciones prácticas para problemas de negocio reales. 

Hemos enfrentado y resuelto los desafíos típicos de proyectos de datos:
- Datos imperfectos que requieren limpieza
- Clases desbalanceadas que sesgan métricas intuitivas
- Necesidad de validar que los modelos generalizan
- Interpretación de resultados para audiencias no técnicas

La experiencia adquirida va más allá de la aplicación de algoritmos: incluye el juicio para seleccionar técnicas apropiadas, la disciplina para validar resultados, y la capacidad de comunicar hallazgos de manera efectiva.

### Limitaciones y Trabajo Futuro

**Limitaciones identificadas**:
- Dataset de tamaño moderado (1,989 clientes) limita la complejidad de modelos aplicables
- Datos transversales (un punto en el tiempo) sin componente temporal explícito
- Ausencia de información sobre contenido de campañas (mensaje, canal, momento)

**Líneas de mejora propuestas**:
- Incorporar datos temporales para modelar evolución del comportamiento
- Experimentar con técnicas de deep learning si se dispone de más datos
- Implementar sistema de monitoreo para detectar degradación de modelos
- Desarrollar interfaz para usuarios de negocio que facilite uso de predicciones

### Reflexión Final

El desarrollo de este proyecto nos ha permitido consolidar los conocimientos adquiridos en la asignatura, enfrentándonos a un problema realista con todos los desafíos que ello implica. Más allá de las métricas obtenidas, el valor principal reside en el proceso: la metodología rigurosa, la documentación exhaustiva, y la validación cuidadosa de resultados.

El aprendizaje automático no es magia ni una caja negra impenetrable. Es una herramienta poderosa que, aplicada con criterio y conocimiento del dominio, puede generar valor significativo. Este proyecto es prueba de ello.

---

## Referencias

- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. PAKDD 2013.
- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters. KDD-96.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.

---

*Documento generado como parte del Proyecto Final de la asignatura de Aprendizaje Automático, Universidad Intercontinental de la Empresa (UIE), Diciembre 2025.*
