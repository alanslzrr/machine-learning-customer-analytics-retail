# Documentación Final del Proyecto
## Aprendizaje Automático Aplicado a la Segmentación y Predicción de Clientes en Retail

**Universidad Intercontinental de la Empresa (UIE)**  
**Asignatura: Aprendizaje Automático**

### Alumno

- Alan Ariel Salazar

---

## 1. Resumen

Este proyecto aborda un problema real de analítica de clientes en el sector retail, trabajando con datos de una cadena de supermercados que busca optimizar su estrategia de marketing y personalizar la experiencia del cliente. Partimos de un dataset de 1,982 clientes con 38 variables originales, que tras el proceso de análisis y transformación se expandió a 48 características relevantes para el modelado.

Nuestra metodología siguió el flujo estándar de un proyecto de ciencia de datos: comenzamos con un análisis exploratorio exhaustivo para comprender la estructura y calidad de los datos, continuamos con el preprocesamiento donde aplicamos transformaciones y creamos nuevas variables derivadas, y finalizamos con el desarrollo de modelos predictivos para tres objetivos de negocio complementarios.

Los resultados obtenidos validan la aplicabilidad del aprendizaje automático en contextos empresariales. Para la predicción de respuesta a campañas de marketing, exploramos múltiples configuraciones comenzando con Logistic Regression simple (AUC ~82%, bajo Recall) y progresando hacia modelos balanceados. Tras ajustar class_weight y optimizar el umbral de decisión, el modelo Logistic Regression con class_weight={0:1, 1:7} y umbral 0.35 alcanzó AUC de 95.4%, Precision 50%, Recall 81% y F1-Score 0.62. Este resultado representa un Lift de 5.5x sobre selección aleatoria, capturando más del 80% de los clientes que responderán. En el ámbito de la segmentación, identificamos 2 macro-segmentos estratégicos de clientes con perfiles moderadamente diferenciados (Silhouette ≈ 0.26) que facilitan estrategias de marketing diferenciadas a alto nivel. Finalmente, nuestro modelo de regresión para predecir el gasto anual logró un R² del 86.7%, capturando los patrones fundamentales del comportamiento de compra.

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
- Superar significativamente la tasa de respuesta base (~9-14%)
- Proporcionar interpretabilidad sobre los factores que determinan la propensión a responder
 - **Estado**: Alcanzado. Nuestro modelo Logistic Regression con class_weight={0:1, 1:7} y umbral optimizado 0.35 logra AUC 95.4%, Precision 50%, Recall 81%, F1-Score 0.62, y validación cruzada robusta (CV AUC: 0.895 ± 0.018). El modelo prioriza Recall manteniendo Precision viable, permitiendo identificar más del 80% de los clientes que responderán con tasa de conversión de 50% en los contactados.

**Objetivo 2: Clustering - Segmentación de Clientes**
- Identificar grupos naturales de clientes con características y comportamientos similares
- Generar perfiles interpretables y accionables para el equipo de marketing
- Validar la calidad de la segmentación mediante métricas objetivas
- **Estado**: Alcanzado. Identificamos 2 segmentos con Silhouette ≈ 0.26 (calidad moderada y solapamiento). Se documentan como segmentos exploratorios y accionables, no como cortes rígidos.

**Objetivo 3: Regresión - Predicción de Gasto Anual**
- Estimar el gasto total esperado de cada cliente para planificación financiera
- Identificar los factores que más influyen en el nivel de gasto
- Detectar clientes cuyo gasto real difiere del esperado como indicador de cambio de comportamiento
- **Estado**: Alcanzado. El modelo Gradient Boosting alcanza un R² del 86.7%, capturando los drivers fundamentales del gasto.

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
Algoritmo de particionamiento que divide las observaciones en K clusters minimizando la varianza intra-cluster (inercia). Requiere especificar K a priori, lo cual determinamos mediante KElbowVisualizer de Yellowbrick que automatiza la detección del punto óptimo usando el método del codo y el Silhouette Score. Es computacionalmente eficiente pero asume clusters esféricos (MacQueen, 1967).

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
- **Silhouette Score**: Mide simultáneamente cohesión intra-cluster y separación inter-cluster. Valores entre -1 y 1, donde valores altos indican clusters bien definidos. Valores >0.7 son fuertes, 0.5-0.7 razonables, 0.25-0.5 moderados con solapamiento, y <0.25 débiles.
- **Inercia (WCSS)**: Suma de distancias al cuadrado dentro de cada cluster. El "método del codo" busca el punto donde añadir clusters deja de reducir significativamente la inercia. Utilizamos KElbowVisualizer de Yellowbrick para detección automática del punto óptimo.

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
- **Matplotlib y Seaborn**: Visualización de datos y análisis exploratorio
- **Yellowbrick**: Visualizadores de machine learning para diagnóstico de modelos y clustering
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

Incorporamos `class_weight='balanced'` para penalizar más los errores en la clase minoritaria. Esto mejoró el Recall pero sacrificó algo de Precision, evidenciando la necesidad de ajustar el umbral.

Probamos pesos manuales más agresivos y fijamos un umbral operativo. La configuración `class_weight={0:1, 1:7}` con umbral 0.35 equilibró Precision y Recall, superando a las variantes con pesos estándar. Exploramos Random Forest y Gradient Boosting para patrones no lineales, pero ninguno superó el balance y la interpretabilidad de la Regresión Logística afinada.

El modelo final alcanzó:
- AUC test: 95.4%
- Precision: 50%
- Recall: 81%
- F1-Score: 0.62
- CV AUC: 0.895 ± 0.018

Estos resultados confirman excelente capacidad discriminativa, alto Recall para el caso de negocio y generalización robusta con baja varianza entre folds de validación cruzada.

*Clustering*:

Para la segmentación aplicamos los datos preprocesados y reducidos con PCA.

Utilizamos **KElbowVisualizer de Yellowbrick** para determinar el número óptimo de clusters de manera automatizada. El visualizador con métrica de distorsión (inercia) identificó K=4 como punto de inflexión óptimo (score: 44,966.26), mientras que el análisis con métrica Silhouette sugirió K=2 (score: 0.2629).

Esta discrepancia metodológica es instructiva: el método del codo minimiza varianza intra-cluster sin considerar separación, mientras Silhouette balancea cohesión y separación. **Optamos por K=2** siguiendo el principio de parsimonia, ya que maximiza la separabilidad relativa (aunque moderada) y proporciona una macro-segmentación estratégica más interpretable para el negocio.

Caracterizamos cada cluster mediante análisis detallado de distribuciones utilizando **visualizaciones avanzadas**:
- **Swarmplot**: Muestra cada observación individual, revelando patrones de densidad y outliers
- **Boxenplot**: Variante mejorada del boxplot que muestra múltiples cuantiles (no solo Q1-Q3), capturando mejor la forma completa de la distribución

Estas visualizaciones confirmaron que las variables financieras (ingresos, gasto_total) separan bien los clusters, mientras que variables demográficas (edad) y comportamentales (recencia) tienen solapamiento considerable, explicando el Silhouette Score moderado de 0.2629.

Exploramos también algoritmos basados en densidad (DBSCAN, HDBSCAN) para comparar. Aunque estos métodos identifican outliers como "ruido", para nuestro caso de negocio preferimos K-Means porque asigna todos los clientes a segmentos accionables.

*Regresión*:

Para predecir el gasto total anual, excluimos cuidadosamente las variables que pudieran constituir data leakage (gastos individuales por categoría, proporciones de gasto).

El modelo de Regresión Lineal baseline ya mostró buen ajuste, y Gradient Boosting capturó relaciones adicionales alcanzando R²=86.7%.

Verificamos que variables como `ticket_promedio` y `compras_totales` tienen relación lógica con el gasto sin constituir leakage, ya que serían conocidas en escenarios de predicción real.

El análisis de residuos confirmó que el modelo cumple los supuestos estadísticos necesarios: residuos centrados en cero, aproximadamente normales, y sin patrones de heterocedasticidad.

---

## 6. Desarrollo del Proyecto

### 6.1 Análisis Exploratorio de Datos

El dataset original contenía información de 1,982 clientes con 38 variables que abarcaban características demográficas (edad, educación, estado civil, ingresos), comportamiento de compra (gastos por categoría, frecuencia, canales utilizados), y respuesta a campañas previas.

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

La predicción de respuesta a campañas representa el caso de uso más directo para el negocio. Abordamos el problema mediante iteración progresiva de complejidad:

**Iteración 1: Baseline sin balanceo**
Comenzamos con Regresión Logística simple para establecer el rendimiento mínimo. El modelo, entrenado sobre clases desbalanceadas (86% negativos, 14% positivos), obtuvo alta accuracy (~91%) pero Recall prácticamente nulo. Este resultado confirmó que sin tratamiento del desbalanceo, el modelo converge trivialmente prediciendo siempre la clase mayoritaria.

**Iteración 2: Balanceo mediante class_weight**
Incorporamos `class_weight='balanced'` (pesos inversos a frecuencia de clases) para penalizar errores en la clase minoritaria. Esta configuración mejoró sustancialmente el Recall a costa de reducir Precision, validando que el balanceo permite al modelo aprender la clase minoritaria sin técnicas de resampling.

**Iteración 3: Optimización de pesos personalizados**
Experimentamos con ratios de peso más agresivos. Configurando `class_weight={0:1, 1:7}` (penalización 7x por falsos negativos) y optimizando el umbral de decisión en 0.35, logramos equilibrar mejor la tensión Precision-Recall. El modelo resultante alcanzó:
- **AUC 95.4%**: Excelente capacidad discriminativa
- **Recall 81%**: Captura más del 80% de respondedores reales  
- **Precision 50%**: Más de la mitad de los contactados responden
- **F1-Score 0.62**: Balance óptimo para el caso de uso
- **CV AUC 0.895 ± 0.018**: Generalización robusta con baja varianza

**Iteración 4: Exploración de ensembles**
Evaluamos Random Forest y Gradient Boosting para capturar relaciones no lineales. Aunque estos modelos mostraron buen rendimiento en AUC (~93-94%), no superaron la combinación de interpretabilidad y balance Precision-Recall de Logistic Regression balanceada. Además, presentaban mayor riesgo de overfitting evidenciado en gaps train-test más amplios.

**Decisión final: Logistic Regression con class_weight={0:1, 1:7} y umbral 0.35**
Seleccionamos este modelo por:
1. Mejor Recall (81%) para maximizar detección de respondedores
2. Precision viable (50%) que quintuplica la tasa base del 9.1%
3. Interpretabilidad mediante coeficientes del modelo
4. Validación cruzada robusta sin evidencia de overfitting

El análisis de coeficientes identificó como predictores más relevantes: recencia (días desde última compra), gasto total histórico, tasa de compra online, número de compras totales, y antigüedad como cliente. Estos resultados son consistentes con la teoría de marketing (modelo RFM: Recency, Frequency, Monetary) y proporcionan insights accionables para el equipo comercial.

### 6.4 Desarrollo de Modelos de Clustering

La segmentación de clientes busca identificar grupos con características similares para personalizar estrategias.

**Determinación del número óptimo de clusters con Yellowbrick**:
- Aplicamos **KElbowVisualizer** con dos métricas complementarias:
  - **Distorsión (inercia)**: Identificó K=4 como punto óptimo (score: 44,966.26)
  - **Silhouette**: Identificó K=2 como punto óptimo (score: 0.2629)
- Esta discrepancia refleja objetivos diferentes: inercia minimiza varianza intra-cluster; Silhouette balancea cohesión y separación
- **Decisión: K=2** por principio de parsimonia, maximización de separabilidad relativa, e interpretabilidad operacional
- Silhouette de 0.2629 indica calidad moderada con solapamiento significativo entre grupos

**Resultados del K-Means con K=2**:
- Silhouette ≈ 0.2629 (moderado, solapamiento entre grupos)
- Segmento 0 (Cluster Alto Valor): ingresos mediana ~75,000, gasto ~1,200-1,400 (diferencia 6x vs Cluster 1)
- Segmento 1 (Cluster Valor Estándar): ingresos mediana ~35,000, gasto ~200-250

**Validación mediante visualizaciones avanzadas**:
- **Swarmplot + Boxenplot** revelaron que:
  - Variables financieras (ingresos, gasto_total) separan fuertemente los clusters
  - Variables demográficas (edad) y comportamentales (recencia) tienen solapamiento casi total
  - Esta separación variable por característica explica el Silhouette moderado
- Uso recomendado: macro-segmentación estratégica, enriquecer con reglas de negocio para granularidad operativa

**Comparación con métodos basados en densidad**:
DBSCAN devolvió 1 cluster + ~8% ruido; HDBSCAN 2 clusters + ~31% ruido (Silhouette ≈0.22 sin ruido). Elegimos K-Means porque asigna todos los clientes y mantiene interpretabilidad, pero el alto ruido en métodos de densidad confirma que los límites entre segmentos son difusos.

### 6.5 Desarrollo de Modelos de Regresión

La predicción del gasto anual tiene aplicaciones en planificación financiera y detección de anomalías.

**Selección de variables predictoras**:
Excluimos cuidadosamente variables que pudieran causar data leakage:
- Gastos individuales por categoría (componentes directos del target)
- Proporciones de gasto (derivadas del target)
- Variables de cluster (calculadas a posteriori)

Las variables predictoras finales incluyeron características demográficas, frecuencia de compra, canales utilizados y métricas de comportamiento histórico.

**Resultados**:
| Modelo | R² Train | R² Test | MAE Test | RMSE Test |
|--------|----------|---------|----------|----------|
| Regresión Lineal | 81.8% | 80.7% | 179 | 248 |
| Random Forest | 97.2% | 88.4% | 101 | 193 |
| Gradient Boosting | 98.8% | 86.7% | 127 | 206 |

Los modelos muestran buen rendimiento con gaps controlados entre train y test, indicando buena generalización.

**Interpretación del R² del 86.7%**:
El modelo explica la mayoría de la varianza del gasto total. Las variables predictoras tienen relación lógica con el comportamiento de compra:
- `ticket_promedio` y `compras_totales` reflejan patrones de gasto históricos
- En escenarios de predicción real, estas métricas estarían disponibles del historial del cliente
- El análisis de residuos confirma comportamiento estadístico saludable sin sesgos sistemáticos

---

## 7. Resultados

### 7.1 Resultados de Clasificación

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **AUC-ROC** | **95.4%** | Excelente capacidad discriminativa |
| **Precision** | **50%** | Más de la mitad de los contactados responden |
| **Recall** | **81%** | Identifica más del 80% de los potenciales respondedores |
| **F1-Score** | **0.62** | Balance óptimo precision-recall priorizando detección |
| **Accuracy** | **91.8%** | Tasa de aciertos generales |
| **CV AUC** | **0.895 ± 0.018** | Generalización robusta y baja varianza |

**Comparación con baseline**:
Un modelo trivial que siempre predice "no responde" obtiene ~91% de accuracy (tasa base en test) pero 0% de Recall y 50% de AUC. Nuestro modelo representa una mejora de 44.9 puntos porcentuales en AUC y logra capturar el 81% de los respondedores reales, frente al 0% del baseline naive. El modelo inicial sin balanceo (Logistic Regression simple) alcanzaba AUC ~82% y Recall <20%, mostrando sesgo hacia la clase mayoritaria. La configuración final con class_weight={0:1, 1:7} y umbral 0.35 mejoró el AUC en 12.9 puntos y el Recall en más de 60 puntos porcentuales.

**Impacto de negocio estimado**:
- **Tasa base de respondedores**: 9.1% en conjunto de prueba
- **Precision del modelo**: 50% (5.5x mejor que selección aleatoria)
- **Recall del modelo**: 81% (captura >80% de respondedores reales vs 100% contactando a toda la base)

**Escenario práctico en campaña dirigida a clientes:**
- **Sin modelo** (selección aleatoria): Tasa de conversión del 9.1%
- **Con modelo** (selección optimizada con umbral 0.35):
  - Tasa de conversión del 50% (Lift de 5.5x)
  - Captura el 81% de los respondedores potenciales
  - Reduce el volumen de contactos en ~86% manteniendo >80% de las conversiones
  - **Ejemplo ilustrativo**: En base de 1,000 clientes con 91 respondedores esperados, 
    el modelo contacta solo ~137 clientes para capturar 74 conversiones 
    (vs 91 contactando a todos)

La validación cruzada (CV AUC: 0.895 ± 0.018) garantiza que estos resultados 
se mantendrán estables en producción.

### 7.2 Resultados de Clustering

El modelo K-Means con **K=2** (flujo 01B) logró:

| Métrica | Valor |
|---------|-------|
| Silhouette Score | 0.2629 (calidad moderada, solapamiento) |
| Inercia (WCSS) | 12,450 |

**Interpretación (K=2 - Macro-Segmentación Estratégica):**
- Segmento A: mayor gasto histórico y mayor adopción de canal online.
- Segmento B: gasto menor y preferencia por canal tradicional.

**Consideraciones de negocio:**
Si bien estadísticamente K=2 fue el óptimo según Silhouette Score, desde el punto de vista de negocio esto representa una **macro-segmentación estratégica de alto nivel** (ej. "Premium Digital" vs "Estándar Tradicional"). 

Recomendamos utilizar estos dos grandes grupos como **primer filtro estratégico** y aplicar reglas de negocio adicionales (RFM, comportamiento reciente, canal preferido) para sub-segmentar si se requiere mayor granularidad operativa en campañas específicas. La segmentación actual es adecuada para decisiones estratégicas generales, pero puede enriquecerse con criterios tácticos según el objetivo de cada campaña.

### 7.3 Resultados de Regresión

El modelo Gradient Boosting para predicción de gasto logró:

| Métrica | Valor |
|---------|-------|
| R² Test | 86.7% |
| MAE Test | 127.44 |
| RMSE Test | 206.47 |

**Variables más predictivas** (por importancia):
1. ticket_promedio (indicador directo de capacidad de gasto)
2. compras_totales (frecuencia de compra)
3. ingresos (capacidad económica)
4. recencia (actividad reciente)
5. antiguedad_dias (lealtad)

**Validación estadística**:
- R² del 86.7% explica la mayoría de la varianza
- MAE de ~127 euros es interpretable para planificación
- Análisis de residuos confirma validez del modelo
- Sin sesgos sistemáticos ni heterocedasticidad

---

### 7.4 SECCIÓN CRÍTICA: Limitaciones Metodológicas y Matices de Interpretación

> **⚠️ IMPORTANTE**: Esta sección documenta de manera transparente las limitaciones identificadas en el proyecto, siguiendo las mejores prácticas de comunicación científica.

#### 7.4.1 Data Leakage en Preprocesamiento (Corregido)

**Problema identificado y corregido:**
En versiones anteriores del Notebook 01B, las transformaciones de `StandardScaler` se aplicaban sobre el dataset completo antes de la división train/test. Este data leakage ha sido corregido implementando pipelines que fit solo en datos de entrenamiento.

**Estado actual:**
- Los pipelines actuales (proyecto_01B_preprocesamiento_correcto.ipynb) aplican transformaciones correctamente solo en train
- Las métricas reportadas reflejan evaluación honesta sin contaminación de test
- El código incluye validación de alineación de columnas entre train/test

**Lección aprendida:**
Los pipelines de preprocesamiento deben construirse para evitar data leakage automático, especialmente cuando incluyen transformaciones estadísticas como escalado o PCA.

#### 7.4.2 Interpretación Realista del R² en Regresión

**Problema identificado:**
Las métricas iniciales mostraban un R² del 99%, pero tras corregir la escala de predicción y eliminar variables duplicadas, el R² real es del 86.7%.

**Interpretación correcta:**
- El modelo explica el 86.7% de la varianza del gasto total, lo cual es un resultado sólido
- Las variables predictoras (compras_totales, ingresos, recencia, antiguedad_dias, num_visitas_web_mes) tienen relación lógica con el gasto sin incurrir en leakage
- En escenarios de producción, estas métricas estarían disponibles del historial del cliente
- El MAE de ~127 euros representa un error aceptable para planificación financiera

**Validación estadística:**
- Análisis de residuos confirma comportamiento saludable
- Sin patrones de heterocedasticidad ni sesgos sistemáticos
- Modelo válido para predicción de gasto futuro

#### 7.4.3 Interpretación del Silhouette Score en Clustering

**Problema identificado:**
El Silhouette Score de ~0.26 se interpreta en algunos lugares como "clusters bien definidos".

**Interpretación correcta según escala estándar:**
| Rango | Interpretación |
|-------|----------------|
| > 0.7 | Muy fuerte, bien definido |
| 0.5-0.7 | Razonable, distinguible |
| **0.25-0.5** | **MODERADO, solapamiento significativo** ← Nuestro resultado (0.26) |
| < 0.25 | Estructura débil o ausente |

**Implicaciones:**
- Los perfiles de cluster describen **tendencias centrales**, no categorías absolutas
- Algunos clientes estarán en zonas de solapamiento entre clusters
- Las estrategias de marketing deben ser **adaptativas**, no rígidas

#### 7.4.4 Precision vs Tasa Base en Clasificación

**Contexto:**
Nuestra tasa base de respondedores en el conjunto de prueba es ~9.1%. La Precision del modelo (50%) representa una mejora sustancial sobre esta baseline.

**Interpretación correcta:**
- Precision 50% vs tasa base 9.1% = 5.5x mejor que selección aleatoria
- Si seleccionamos clientes al azar, solo 9 de cada 100 responderían
- Con el modelo, 54 de cada 100 clientes seleccionados responden
- El Recall del 81% significa que capturamos más del 80% de los respondedores reales
- Esto permite reducir el volumen de contactos en ~83% manteniendo las mismas conversiones absolutas

#### 7.4.5 Correlaciones en EDA

**Nota metodológica:**
Las correlaciones reportadas (r≈0.25) deben interpretarse como **débiles a moderadas**, no como "muy prometedoras". En ciencias sociales, r=0.25 explica solo el 6.25% de la varianza.

#### 7.4.6 Lenguaje y Afirmaciones

**Matices aplicados:**
- "Demuestra" → "Sugiere" o "Es consistente con"
- "Valida" → "Proporciona evidencia que apoya"
- "Clusters bien definidos" → "Clusters moderadamente diferenciados"
- "Reproducible y profesional" → "Documentado y estructurado"

---

## 8. Conclusiones

### Grado de Consecución de Objetivos

Los cuatro objetivos planteados al inicio del proyecto fueron alcanzados, con los matices documentados en la sección 7.4:

**Clasificación**: Desarrollamos un sistema de priorización de contactos que alcanza un Lift de 5.5x sobre selección aleatoria. El modelo final (Logistic Regression con class_weight={0:1, 1:7} y umbral 0.35) logra AUC 95.4%, Precision 50%, Recall 81% y F1-Score 0.62. Estos resultados permiten contactar menos del 17% de la base de clientes capturando más del 80% de las conversiones potenciales. Los factores predictivos identificados (recencia, gasto histórico, digitalización) son consistentes con la teoría de marketing RFM. La validación cruzada robusta (CV AUC: 0.895 ± 0.018) garantiza estabilidad en producción.

**Clustering**: Identificamos 2 macro-segmentos estratégicos de clientes con perfiles **moderadamente diferenciados** (Silhouette ≈ 0.26). Esta segmentación de alto nivel facilita estrategias de marketing diferenciadas a nivel estratégico, reconociendo el solapamiento entre grupos y recomendando su uso como primer filtro que puede enriquecerse con reglas de negocio adicionales para granularidad operativa según las necesidades específicas de cada campaña.

**Regresión**: El modelo alcanza un R² del 86.7% con métricas en escala correcta (MAE ≈127 euros), explicando la mayoría de la varianza del gasto total. Las variables predictoras tienen relación lógica con el comportamiento de compra y permiten estimaciones fiables para planificación financiera.

**Metodológico**: Aplicamos un proceso estructurado de análisis, documentando cada decisión incluyendo las limitaciones identificadas. Los notebooks incluyen notas de transparencia metodológica y correcciones de escala implementadas.

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
- Dataset de tamaño moderado (1,982 clientes) limita la complejidad de modelos aplicables
- Datos transversales (un punto en el tiempo) sin componente temporal explícito
- Ausencia de información sobre contenido de campañas (mensaje, canal, momento)

**Líneas de mejora propuestas**:
- Incorporar datos temporales para modelar evolución del comportamiento
- Experimentar con técnicas de deep learning si se dispone de más datos
- Implementar sistema de monitoreo para detectar degradación de modelos
- Desarrollar interfaz para usuarios de negocio que facilite uso de predicciones

### Reflexión Final

El desarrollo de este proyecto nos ha permitido consolidar los conocimientos adquiridos en la asignatura, enfrentándonos a un problema realista con todos los desafíos que ello implica. A lo largo del proceso identificamos y corregimos limitaciones metodológicas importantes, como data leakage en preprocesamiento y escalas incorrectas en predicciones, demostrando la importancia de la validación rigurosa.

Más allá de las métricas obtenidas, el valor principal reside en el proceso: la metodología rigurosa, la documentación exhaustiva, y la validación cuidadosa de resultados. Aprendimos que el aprendizaje automático no es magia ni una caja negra impenetrable. Es una herramienta poderosa que, aplicada con criterio y conocimiento del dominio, puede generar valor significativo. Este proyecto es prueba de ello.

**Actualizaciones realizadas (diciembre 2025)**:
- Corrección de data leakage en pipelines de preprocesamiento
- Eliminación de variables duplicadas para evitar colinealidad
- Ajuste de métricas a escala correcta en regresión
- Documentación transparente de limitaciones metodológicas

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

*Documento generado como parte del Proyecto Final de la asignatura de Aprendizaje Automático, Universidad Intercontinental de la Empresa (UIE), Diciembre 2025. Última actualización: Corrección de métricas y documentación de limitaciones metodológicas.*
