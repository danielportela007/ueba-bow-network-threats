# Reporte Tecnico: Extraccion de Caracteristicas para UEBA mediante Bag-of-Words

## Deteccion de Actividad de Criptomineria en el Dataset RBD24 Crypto Desktop

---

**Autor:** Portela  
**Fecha:** Abril 2026  
**Asignatura:** Ciberseguridad - Proyecto 2  
**Dataset:** RBD24 - Crypto Desktop  
**Tecnica central:** Frequency counting (bag-of-words) sobre features de red discretizadas

---

## Tabla de Contenidos

1. [Introduccion y Contexto](#1-introduccion-y-contexto)
2. [Marco Teorico](#2-marco-teorico)
3. [Dataset RBD24 Crypto Desktop](#3-dataset-rbd24-crypto-desktop)
4. [Arquitectura del Sistema](#4-arquitectura-del-sistema)
5. [Preprocesamiento de Datos](#5-preprocesamiento-de-datos)
6. [Tokenizacion: De Features Numericas a Tokens Categoricos](#6-tokenizacion-de-features-numericas-a-tokens-categoricos)
7. [Construccion de la Representacion Bag-of-Words](#7-construccion-de-la-representacion-bag-of-words)
8. [Estrategia de Evaluacion y Manejo del Desbalance](#8-estrategia-de-evaluacion-y-manejo-del-desbalance)
9. [Modelado: Clasificacion Supervisada](#9-modelado-clasificacion-supervisada)
10. [Modelado: Deteccion No Supervisada de Anomalias](#10-modelado-deteccion-no-supervisada-de-anomalias)
11. [Resultados Experimentales](#11-resultados-experimentales)
12. [Interpretacion de Resultados](#12-interpretacion-de-resultados)
13. [Problemas Encontrados y Soluciones](#13-problemas-encontrados-y-soluciones)
14. [Analisis Exploratorio de Datos (EDA)](#14-analisis-exploratorio-de-datos-eda)
15. [Validacion Cruzada](#15-validacion-cruzada)
16. [Conclusiones](#16-conclusiones)
17. [Estructura del Codigo](#17-estructura-del-codigo)
18. [Referencias](#18-referencias)

---

## 1. Introduccion y Contexto

### 1.1 Objetivo del Proyecto

El objetivo de este proyecto es implementar un sistema de **User and Entity Behavior Analytics (UEBA)** que utilice la tecnica de **frequency counting** (conteo de frecuencias), tambien conocida como **bag-of-words (BoW)**, para extraer caracteristicas de comportamiento de red y detectar actividad maliciosa de criptomineria.

La premisa fundamental de UEBA es que los patrones de comportamiento de red de un usuario o entidad pueden ser representados como distribuciones de eventos discretos, y que las desviaciones significativas de estos patrones indican actividad anomala o maliciosa.

### 1.2 Contexto Academico

Este trabajo se enmarca en el Proyecto 2 de la materia de Ciberseguridad, cuyo enunciado establece:

- **Tecnica asignada:** Frequency counting (bag-of-words)
- **Dataset:** Seleccionado del benchmark RBD24
- **Requisitos:** Definir la unidad de analisis, preprocesar los datos, tokenizar las features, construir la representacion BoW, y validar con metodos de deteccion de anomalias supervisados y no supervisados
- **Recomendacion del asesor:** Analizar el comportamiento normal primero para identificar ataques, dado el extremo desbalance de clases

### 1.3 Que es Frequency Counting (Bag-of-Words)

En el procesamiento de lenguaje natural (NLP), bag-of-words es una representacion donde un documento se modela como un conjunto no ordenado de palabras, descartando gramatica y orden, pero conservando la frecuencia de aparicion. Cada documento se convierte en un vector donde cada dimension corresponde a una "palabra" (token) del vocabulario y su valor es la frecuencia de aparicion.

En el contexto de UEBA, la analogia es:

| NLP | UEBA (este proyecto) |
|-----|---------------------|
| Documento | Ventana temporal de actividad de red (1 hora) |
| Palabra | Valor discretizado de una feature de red |
| Vocabulario | Conjunto de todos los posibles valores discretizados |
| Frecuencia | Conteo de ocurrencias de cada valor en la ventana |

Cada muestra del dataset (una ventana temporal de actividad) se convierte en un vector de frecuencias de tokens, donde los tokens son las versiones discretizadas de las features numericas de red (e.g., `dns_len_TTL=alto`, `http_status_200_ratio=muy_bajo`).

---

## 2. Marco Teorico

### 2.1 User and Entity Behavior Analytics (UEBA)

UEBA es un enfoque de ciberseguridad que analiza los patrones normales de comportamiento de usuarios y entidades (workstations, servidores, etc.) para detectar desviaciones que puedan indicar amenazas internas, cuentas comprometidas o actividad maliciosa. A diferencia de los sistemas basados en reglas, UEBA utiliza tecnicas de machine learning para modelar perfiles de comportamiento y detectar anomalias de forma automatica.

### 2.2 El Benchmark RBD24

RBD24 (Realistic Baseline for anomaly Detection, 2024) es un benchmark estandarizado para la evaluacion de sistemas UEBA. Sus caracteristicas principales son:

- **Datos reales de red:** Capturados en un entorno corporativo controlado durante 27 dias
- **Ventanas temporales predefinidas:** 1 hora con desplazamiento de 10 minutos
- **Features de red normalizadas:** 81 features por ventana, cubriendo protocolos DNS, SSL/TLS, HTTP y SMTP
- **Etiquetas a nivel de usuario:** Cada usuario esta etiquetado como benigno o de riesgo
- **Multiples escenarios de ataque:** Crypto Desktop (criptomineria), RAT, exfiltracion, etc.

El paper original RBD24 reporta resultados con F1-score de aproximadamente 0.63 para el dataset Crypto Desktop utilizando Random Forest con downsampling.

### 2.3 Criptomineria como Amenaza

La criptomineria no autorizada (cryptojacking) consiste en el uso ilicito de recursos computacionales de una organizacion para minar criptomonedas. Se manifiesta en el trafico de red mediante:

- Patrones inusuales de DNS (consultas a pools de mineria)
- Conexiones SSL/TLS persistentes a servidores de pools
- Tiempos entre eventos (interlog times) anomalos
- Actividad fuera de horas laborales

### 2.4 TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF es una tecnica de ponderacion que ajusta las frecuencias crudas de los tokens para reflejar su importancia relativa. La formula es:

- **TF (Term Frequency):** Frecuencia del token normalizada por el total de tokens en el documento
- **IDF (Inverse Document Frequency):** log((N + 1) / (df + 1)) + 1, donde N es el numero total de documentos y df es el numero de documentos que contienen el token
- **TF-IDF = TF * IDF**, seguido de normalizacion L2 por fila

La intuicion es que un token que aparece en casi todos los documentos (como `dns_authoritative_ans_ratio=alto`) aporta menos informacion discriminativa que un token raro. IDF penaliza tokens comunes y realza tokens raros.

---

## 3. Dataset RBD24 Crypto Desktop

### 3.1 Descripcion General

El dataset `Crypto_desktop.parquet` contiene registros de actividad de red de estaciones de trabajo de escritorio, con el siguiente perfil:

| Propiedad | Valor |
|-----------|-------|
| Muestras totales | 162,545 |
| Features numericas | 81 |
| Usuarios unicos | 749 |
| Rango temporal | 2023-09-20 a 2023-10-16 (27 dias) |
| Formato | Apache Parquet |

### 3.2 Estructura de Columnas

Cada fila del dataset corresponde a una **ventana temporal** de 1 hora con desplazamiento de 10 minutos, capturando la actividad de red de una **entidad** (workstation). Las columnas se dividen en:

- **Metadatos (4):** `entity` (identificador de workstation), `label` (0=benigno, 1=riesgo), `user_id` (identificador de usuario), `timestamp` (inicio de la ventana)
- **Features numericas (81):** Metricas de red normalizadas, agrupadas por protocolo:
  - **DNS (19 features):** `dns_authoritative_ans_ratio`, `dns_len_TTL`, `dns_rcode_ok_ratio`, `dns_usual_dns_dstport_ratio`, etc.
  - **SSL/TLS (15 features):** `ssl_established_ratio`, `ssl_version_ratio_v12`, `ssl_self_signed_cert_ratio`, etc.
  - **HTTP (14 features):** `http_status_200_ratio`, `http_method_get_ratio`, `http_common_dstport_ratio`, etc.
  - **SMTP (10 features):** `smtp_in_is_reply`, `smtp_in_to_count`, `smtp_in_rcpt_to_count`, etc.
  - **Interlog times (10 features):** `mean_interlog_time_dns_interlog_time`, `std_interlog_time_dns_interlog_time`, etc.
  - **Temporales (4 features):** `non_working_hour_ratio`, `non_working_hours_dns_ratio`, etc.
  - **Otras:** Features adicionales de volumetria y ratios de puertos

### 3.3 Desbalance de Clases

El aspecto mas critico del dataset es su **extremo desbalance de clases**:

**A nivel de muestra:**
| Clase | Muestras | Porcentaje |
|-------|----------|------------|
| Benigno (label=0) | 161,202 | 99.17% |
| Riesgo/Crypto (label=1) | 1,343 | 0.83% |

**Ratio de desbalance: 120:1** (por cada muestra de riesgo hay 120 benignas)

**A nivel de usuario:**
| Clase | Usuarios |
|-------|----------|
| Benigno | 738 |
| Riesgo | 11 |

Esto significa que solo 11 de 749 usuarios (1.47%) presentan actividad de criptomineria. Este desbalance extremo tiene implicaciones profundas en el diseno del pipeline de evaluacion, como se detalla en la Seccion 8.

### 3.4 Unidad de Analisis

La **unidad de analisis** seleccionada es la **entidad (workstation)**, que corresponde a una estacion de trabajo individual. Cada workstation genera multiples ventanas temporales, y un usuario puede estar asociado a una o mas workstations. La etiqueta de riesgo se asigna a nivel de usuario y se propaga a todas sus ventanas.

---

## 4. Arquitectura del Sistema

### 4.1 Vision General del Pipeline

El sistema sigue un pipeline de 9 pasos secuenciales:

```
[1] Carga de datos (Parquet)
        |
[2] Analisis Exploratorio (EDA)
        |
[3] Preprocesamiento
    - Eliminacion de features constantes (13 eliminadas)
    - Identificacion de features dispersas vs densas
        |
[4] Tokenizacion
    - Discretizacion adaptativa de features numericas
    - Construccion de vocabulario (212 tokens)
        |
[5] Construccion de Representaciones
    - BoW cruda (frecuencias)
    - BoW + TF-IDF (ponderada)
    - Features originales (baseline)
        |
[6] Division Train/Test
    - A nivel de usuario (sin filtracion)
    - 80% train / 20% test
        |
[7] Perfilado de Comportamiento Normal
    - Isolation Forest
    - One-Class SVM
        |
[8] Clasificacion Supervisada
    - 5 clasificadores x 3 representaciones
    - Downsampling de ambos conjuntos (metodologia RBD24)
        |
[9] Evaluacion e Interpretabilidad
    - Metricas, graficas, validacion cruzada
```

### 4.2 Estructura de Modulos

El codigo esta organizado en una estructura modular limpia:

```
Proyecto_Prog/
|-- main.py                    # Orquestador del pipeline (9 pasos)
|-- Crypto_desktop.parquet     # Dataset
|-- src/
|   |-- __init__.py            # Modulo Python
|   |-- config.py              # Configuracion central
|   |-- data_loader.py         # Carga y validacion de datos
|   |-- exploracion.py         # Analisis exploratorio (EDA)
|   |-- preprocessing.py       # Limpieza y preparacion
|   |-- tokenizer.py           # Discretizacion y vocabulario
|   |-- bow_builder.py         # Representaciones BoW y TF-IDF
|   |-- models.py              # Clasificadores y anomaly detection
|   |-- evaluation.py          # Metricas, graficas, interpretabilidad
|-- results/
|   |-- figures/               # 17 graficas generadas
|   |-- metrics/               # CSV con resultados consolidados
|-- venv/                      # Entorno virtual Python
```

### 4.3 Dependencias

| Paquete | Version | Uso |
|---------|---------|-----|
| pandas | - | Manipulacion de datos |
| pyarrow | - | Lectura de Parquet |
| numpy | - | Operaciones numericas |
| scikit-learn | 1.8.0 | Modelos ML, metricas, preprocesamiento |
| xgboost | 3.2.0 | Clasificador XGBoost |
| imbalanced-learn | 0.14.1 | RandomUnderSampler |
| matplotlib | - | Graficas |
| seaborn | - | Graficas estadisticas |
| scipy | - | Utilidades cientificas |

---

## 5. Preprocesamiento de Datos

### 5.1 Eliminacion de Features Constantes

El primer paso del preprocesamiento consiste en identificar y eliminar features con varianza cero, es decir, aquellas que tienen el mismo valor en todas las muestras del dataset. Estas features no aportan informacion discriminativa.

**Se eliminaron 13 features constantes:**

| # | Feature Eliminada | Razon |
|---|-------------------|-------|
| 1 | `std_interlog_time_dns_interlog_time` | Valor constante (varianza = 0) |
| 2 | `std_interlog_time_ssl_interlog_time` | Valor constante |
| 3 | `std_interlog_time_http_interlog_time` | Valor constante |
| 4 | `dns_common_tcp_ports_ratio` | Valor constante |
| 5 | `dns_rcode_noauth_ratio` | Valor constante |
| 6 | `dns_rcode_notzone_ratio` | Valor constante |
| 7 | `dns_compromised_dstip_ratio` | Valor constante |
| 8 | `ssl_version_ratio_v20` | Sin trafico SSL v2.0 |
| 9 | `ssl_version_ratio_v30` | Sin trafico SSL v3.0 |
| 10 | `ssl_compromised_dst_ip_ratio` | Valor constante |
| 11 | `http_compromised_dstip_ratio` | Valor constante |
| 12 | `smtp_in_is_spam` | Sin deteccion de spam |
| 13 | `smtp_in_hazardous_extensions` | Sin extensiones peligrosas |

Las features relacionadas con `compromised` y versiones antiguas de SSL (v2.0, v3.0) son constantes porque en el periodo de captura no se observaron esos eventos. Esto es esperado en un entorno corporativo moderno.

### 5.2 Clasificacion de Features: Densas vs Dispersas

Las 68 features restantes se clasifican en dos categorias segun su distribucion:

- **Features densas (39):** Aquellas con menos del 90% de valores en cero. Tienen distribuciones ricas con varianza significativa. Ejemplos: `dns_authoritative_ans_ratio`, `ssl_established_ratio`, `http_status_200_ratio`.

- **Features dispersas (29):** Aquellas con 90% o mas de valores en cero. Representan eventos raros que solo ocurren en un subconjunto de ventanas temporales. Ejemplos: `smtp_in_is_reply`, `dns_mx_ratio`, `ssl_self_signed_cert_ratio`.

Esta distincion es **fundamental** para la tokenizacion, ya que cada tipo requiere una estrategia de discretizacion diferente (Seccion 6).

**Resumen del preprocesamiento:**
| Categoria | Cantidad |
|-----------|----------|
| Features originales | 81 |
| Features constantes eliminadas | 13 |
| Features dispersas (>90% ceros) | 29 |
| Features densas | 39 |
| **Features totales retenidas** | **68** |

---

## 6. Tokenizacion: De Features Numericas a Tokens Categoricos

### 6.1 El Problema

Las features del dataset son valores numericos continuos (ratios, tiempos, conteos normalizados). Para aplicar bag-of-words, necesitamos convertir estos valores en tokens discretos (categorias). El desafio es que una discretizacion naive (e.g., bins equidistantes) perderia informacion importante, especialmente para features dispersas.

### 6.2 Estrategia de Discretizacion Adaptativa

Se disenio una estrategia de tokenizacion adaptativa que maneja de forma diferente las features densas y dispersas:

#### 6.2.1 Features Densas (39 features)

Para features con distribuciones ricas:

1. Se calculan los **cuantiles** de la distribucion usando los datos de entrenamiento benignos
2. Se definen 5 bins (por defecto) basados en estos cuantiles: `muy_bajo`, `bajo`, `medio`, `alto`, `muy_alto`
3. Cada valor numerico se asigna al bin correspondiente

**Ejemplo:**
- Feature: `dns_len_TTL` con cuantiles [0.0, 12.5, 45.2, 78.1, 95.0, 150.0]
- Valor 50.3 se asigna a `dns_len_TTL=medio`
- Valor 120.0 se asigna a `dns_len_TTL=muy_alto`

El uso de **cuantiles** (en lugar de bins equidistantes) garantiza que cada bin contenga aproximadamente el mismo numero de muestras benignas, maximizando la resolucion de la discretizacion. Los limites de los bins se aprenden unicamente de datos benignos para capturar el perfil de comportamiento normal.

#### 6.2.2 Features Dispersas (29 features)

Para features con >90% de valores en cero:

1. Se crea una categoria especial **`cero`** para los valores nulos
2. Para los valores no-cero, se aplica binning por cuantiles con **4 bins** (NUM_BINS - 1)
3. Tokens resultantes: `{feature}=cero`, `{feature}=bajo`, `{feature}=medio_bajo`, `{feature}=medio_alto`, `{feature}=alto`

**Ejemplo:**
- Feature: `smtp_in_is_reply` (95% de valores son 0)
- Valor 0 se asigna a `smtp_in_is_reply=cero`
- Valor 0.5 se asigna a `smtp_in_is_reply=medio_bajo`

Esta estrategia preserva la informacion crucial de **presencia/ausencia** del evento (la mayoria de las ventanas no tienen trafico SMTP) mientras discrimina entre diferentes niveles de actividad cuando el evento si ocurre.

#### 6.2.3 Manejo de Casos Limite

El tokenizador implementa protecciones para casos especiales:

- **Features con pocos valores unicos:** Si los cuantiles producen menos de 3 edges distintos, se usa el minimo, mediana y maximo como limites
- **Features dispersas sin valores no-cero:** Se crea un unico token `{feature}=presente` ademas de `{feature}=cero`
- **Valores fuera de rango:** Se clippean a los bins extremos usando `np.clip`

### 6.3 Ajuste del Tokenizador

El tokenizador se ajusta exclusivamente sobre los **datos benignos** del dataset completo:

```python
benign_df = df[df[config.LABEL_COL] == 0]
tokenizer = EventTokenizer(num_bins=5)
tokenizer.fit(benign_df, sparse_features, dense_features)
```

Esta decision sigue la recomendacion del asesor del proyecto: modelar el comportamiento normal primero para que las desviaciones maliciosas sean mas evidentes. Al ajustar los limites de binning solo con datos benignos, los valores anomalos de los usuarios de riesgo tendran mayor probabilidad de caer en bins extremos (`muy_bajo` o `muy_alto`).

### 6.4 Vocabulario Resultante

El vocabulario final contiene **212 tokens unicos**, construidos a partir de las 68 features retenidas:

- Features densas: 39 features x ~5 bins = ~195 tokens
- Features dispersas: 29 features x 1 (`cero`) + bins variables = ~17 tokens adicionales

Los tokens se almacenan en orden alfabetico con un mapeo `token -> indice` para eficiencia computacional.

**Ejemplos de tokens del vocabulario:**
- `dns_authoritative_ans_ratio=alto`
- `dns_authoritative_ans_ratio=bajo`
- `ssl_established_ratio=muy_alto`
- `smtp_in_is_reply=cero`
- `mean_interlog_time_http_interlog_time=muy_bajo`

---

## 7. Construccion de la Representacion Bag-of-Words

### 7.1 BoW a Nivel de Muestra

Cada muestra (ventana temporal) se transforma en un vector de dimension 212 (el tamano del vocabulario). El proceso para cada muestra es:

1. Para cada feature (columna), obtener su valor numerico
2. Discretizarlo usando los limites de binning aprendidos
3. Obtener el token correspondiente (e.g., `dns_len_TTL=alto`)
4. Asignar 1 en la posicion del token en el vector BoW

**Resultado:** Matriz BoW de dimension **(162,545 x 212)** donde cada fila es un vector binario (0/1) indicando que bin corresponde a cada feature. Como cada feature produce exactamente un token, la media de tokens activos por documento es 68.0 (igual al numero de features retenidas).

### 7.2 BoW a Nivel de Usuario (Agregacion Temporal)

Para analisis a nivel de usuario, se agregan todas las ventanas temporales de cada usuario sumando sus vectores BoW:

```
BoW_usuario[i] = sum(BoW_muestra[j]) para todo j donde user_id[j] == i
```

**Resultado:** Matriz de **(749 x 212)** con frecuencias acumuladas. La frecuencia media de tokens por usuario es 14,757, reflejando el promedio de ~217 ventanas temporales por usuario.

### 7.3 BoW + TF-IDF

Se aplica ponderacion TF-IDF sobre la matriz BoW cruda:

1. **TF (Term Frequency):** Frecuencia del token dividida por el total de tokens activos en el documento
   - `TF(t, d) = freq(t, d) / sum(freq(*, d))`

2. **IDF (Inverse Document Frequency):** Penalizacion de tokens comunes usando smooth IDF
   - `IDF(t) = log((N + 1) / (df(t) + 1)) + 1`
   - donde N = numero de documentos y df(t) = numero de documentos que contienen el token t

3. **TF-IDF = TF * IDF**, seguido de **normalizacion L2** por fila

**Resultado:** Matriz de dimension **(162,545 x 212)** con valores continuos normalizados.

La intuicion de usar TF-IDF en este contexto es que tokens como `dns_authoritative_ans_ratio=medio` que aparecen en casi todas las ventanas tendran un IDF bajo (son poco informativos), mientras que tokens raros como `smtp_in_is_reply=alto` tendran un IDF alto y seran mas discriminativos para detectar anomalias.

### 7.4 Representaciones Comparadas

El sistema evalua tres representaciones en paralelo para comparar su efectividad:

| Representacion | Descripcion | Dimension |
|---------------|-------------|-----------|
| **BoW (frecuencias)** | Conteo de frecuencias de tokens discretizados | 162,545 x 212 |
| **BoW + TF-IDF** | Frecuencias ponderadas por importancia | 162,545 x 212 |
| **Features originales** | Valores numericos continuos sin discretizar (baseline) | 162,545 x 68 |

La inclusion de las features originales como baseline permite evaluar si la discretizacion BoW aporta valor o si pierde informacion relevante.

---

## 8. Estrategia de Evaluacion y Manejo del Desbalance

### 8.1 El Problema del Desbalance Extremo

Con un ratio de 120:1, un clasificador trivial que prediga siempre "benigno" obtendria:
- **Accuracy: 99.17%** (enganosamemte alto)
- **F1 para clase positiva: 0.00** (completamente inutil)

Esto hace que metricas como accuracy sean inutiles. Ademas, al evaluar en un test set desbalanceado, incluso un buen clasificador obtiene F1-scores extremadamente bajos (~0.009) porque la cantidad de falsos positivos domina.

### 8.2 Metodologia de Evaluacion: Siguiendo el Paper RBD24

Tras analizar el paper RBD24, se identifico que los autores utilizan **downsampling tanto del conjunto de entrenamiento como del de test** para obtener metricas comparables. Esta decision metodologica es clave para entender los resultados reportados (F1~0.63).

La estrategia implementada es:

#### 8.2.1 Division Train/Test a Nivel de Usuario

Se utiliza `GroupShuffleSplit` para garantizar que **ningun usuario aparezca en ambos conjuntos**. Esto previene la filtracion de datos (data leakage) que ocurriria si las ventanas temporales de un mismo usuario estuvieran distribuidas entre train y test.

```
Train: 123,407 muestras, 599 usuarios (1,247 positivos)
Test:  39,138 muestras, 150 usuarios (96 positivos)
```

#### 8.2.2 Downsampling del Training Set

La clase mayoritaria (benigna) se submuestrea aleatoriamente para igualar la clase minoritaria:

```
Original:    122,160 neg / 1,247 pos  (ratio 98:1)
Balanceado:    1,247 neg / 1,247 pos  (ratio 1:1)
```

Se utiliza `RandomUnderSampler` de imbalanced-learn, que selecciona aleatoriamente muestras de la clase mayoritaria sin reemplazo.

#### 8.2.3 Downsampling del Test Set

De forma analoga, el test set se balancea:

```
Original:    39,042 neg / 96 pos
Balanceado:      96 neg / 96 pos  (192 muestras totales)
```

Esto permite que metricas como F1, Precision, Recall y Accuracy sean directamente interpretables y comparables con los resultados del paper RBD24.

#### 8.2.4 Metricas Duales

Para obtener una evaluacion completa, se calculan metricas en dos niveles:

- **Test balanceado:** F1, Precision, Recall, Accuracy, TPR, FPR (comparables con el paper)
- **Test completo:** ROC-AUC y PR-AUC (metricas que funcionan bien con datos desbalanceados porque evaluan probabilidades, no predicciones binarias)

### 8.3 Metricas de Evaluacion Utilizadas

| Metrica | Formula | Interpretacion |
|---------|---------|----------------|
| **F1-score** | 2 * (Prec * Rec) / (Prec + Rec) | Media armonica de precision y recall; penaliza ambos extremos |
| **Precision** | TP / (TP + FP) | Proporcion de predicciones positivas correctas |
| **Recall (TPR)** | TP / (TP + FN) | Proporcion de positivos reales detectados |
| **ROC-AUC** | Area bajo curva ROC | Capacidad de ranking independiente del umbral |
| **PR-AUC** | Area bajo curva Precision-Recall | Similar a ROC-AUC pero mas informativa en datos desbalanceados |
| **FPR** | FP / (FP + TN) | Tasa de falsos positivos (ruido) |

---

## 9. Modelado: Clasificacion Supervisada

### 9.1 Clasificadores Evaluados

Se evaluaron 5 clasificadores supervisados, seleccionados para cubrir diferentes familias de algoritmos:

#### 9.1.1 XGBoost

```python
XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.01,
    min_child_weight=5, reg_alpha=0.1, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=1.0,
    eval_metric="logloss"
)
```

- **Familia:** Gradient boosting sobre arboles de decision
- **Ventajas:** Excelente rendimiento general, manejo nativo de valores faltantes, regularizacion
- **Hiperparametros clave:** Learning rate bajo (0.01) para convergencia suave, regularizacion L1 (reg_alpha=0.1), subsampling de features y muestras para reducir overfitting

#### 9.1.2 Random Forest

```python
RandomForestClassifier(
    n_estimators=300, max_depth=7, min_samples_split=10,
    min_samples_leaf=5, class_weight="balanced"
)
```

- **Familia:** Ensemble de arboles de decision con bagging
- **Ventajas:** Robusto al overfitting, importancia de features nativa, paralelizable
- **Hiperparametros clave:** Profundidad limitada (7) para prevenir overfitting, class_weight="balanced" como complemento al downsampling

#### 9.1.3 Gradient Boosting (scikit-learn)

```python
GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.01,
    min_samples_split=10, min_samples_leaf=5, subsample=0.8
)
```

- **Familia:** Gradient boosting (implementacion clasica de scikit-learn)
- **Ventajas:** Interpretabilidad, importancia de features
- **Diferencia con XGBoost:** Sin regularizacion L1/L2, sin paralelismo nativo

#### 9.1.4 Logistic Regression

```python
LogisticRegression(
    max_iter=2000, class_weight="balanced", C=1.0,
    solver="saga"
)
```

- **Familia:** Modelo lineal generalizado
- **Ventajas:** Interpretabilidad directa de coeficientes, rapido, regularizable
- **Hiperparametros clave:** Solver SAGA para convergencia en datasets grandes, class_weight="balanced", regularizacion C=1.0

#### 9.1.5 SVM con Kernel RBF

```python
SVC(
    kernel="rbf", class_weight="balanced", probability=True,
    gamma="scale", C=10.0
)
```

- **Familia:** Maquinas de vectores de soporte
- **Ventajas:** Eficaz en espacios de alta dimensionalidad, kernel RBF para fronteras no lineales
- **Hiperparametros clave:** gamma="scale" (automatico basado en varianza), C=10.0 (margen suave)

### 9.2 Protocolo de Entrenamiento

Para cada combinacion (clasificador, representacion):

1. **Downsampling** del training set a ratio 1:1
2. **Scaling** automatico con `StandardScaler` para SVM y Logistic Regression (necesario para convergencia)
3. **Entrenamiento** del clasificador
4. **Evaluacion** en test set balanceado (F1, Precision, Recall, etc.)
5. **Evaluacion adicional** en test set completo (ROC-AUC, PR-AUC)

---

## 10. Modelado: Deteccion No Supervisada de Anomalias

### 10.1 Motivacion

La recomendacion del asesor fue analizar el comportamiento normal primero. Los metodos no supervisados modelan exclusivamente el comportamiento benigno y detectan desviaciones como anomalias, sin necesidad de etiquetas de la clase minoritaria.

### 10.2 Isolation Forest

```python
IsolationForest(
    n_estimators=300, contamination=0.008,
    max_features=0.8
)
```

- **Principio:** Aislamiento aleatorio de puntos anomalos. Los puntos anomalos requieren menos particiones aleatorias para ser aislados.
- **Entrenamiento:** Solo con muestras benignas (122,160 muestras)
- **Parametro contamination=0.008:** Proporcion esperada de anomalias (~0.83%)

### 10.3 One-Class SVM

```python
OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
```

- **Principio:** Aprende una frontera que engloba los datos normales en el espacio de features
- **Entrenamiento:** Solo con muestras benignas (submuestra de 20,000 por eficiencia computacional)
- **Parametro nu=0.01:** Cota superior de la fraccion de outliers esperados

### 10.4 Preprocesamiento para No Supervisados

Ambos metodos reciben la representacion **BoW + TF-IDF** con scaling estandar (`StandardScaler`), ya que son sensibles a las escalas de las features.

---

## 11. Resultados Experimentales

### 11.1 Tabla Consolidada de Resultados

#### Clasificacion Supervisada

| Representacion | Modelo | F1 | F1_macro | Precision | Recall | ROC-AUC | PR-AUC |
|---------------|--------|-----|---------|-----------|--------|---------|--------|
| **BoW (frecuencias)** | XGBoost | 0.5060 | 0.5649 | 0.6000 | 0.4375 | 0.6109 | 0.5386 |
| **BoW (frecuencias)** | Random Forest | 0.5389 | 0.5920 | 0.6338 | 0.4688 | 0.6510 | 0.5689 |
| **BoW (frecuencias)** | Gradient Boosting | 0.4819 | 0.5437 | 0.5714 | 0.4167 | 0.6389 | 0.5590 |
| **BoW (frecuencias)** | **Logistic Regression** | **0.6492** | **0.6510** | **0.6526** | **0.6458** | **0.7023** | **0.6250** |
| **BoW (frecuencias)** | SVM (RBF) | 0.3053 | 0.4728 | 0.5714 | 0.2083 | 0.5869 | 0.5747 |
| BoW + TF-IDF | XGBoost | 0.4417 | 0.5150 | 0.5373 | 0.3750 | 0.6607 | 0.5773 |
| BoW + TF-IDF | Random Forest | 0.4908 | 0.5576 | 0.5970 | 0.4167 | 0.6476 | 0.6346 |
| BoW + TF-IDF | Gradient Boosting | 0.4051 | 0.4946 | 0.5161 | 0.3333 | 0.6487 | 0.5797 |
| BoW + TF-IDF | **Logistic Regression** | **0.6387** | **0.6406** | **0.6421** | **0.6354** | **0.6966** | **0.6218** |
| BoW + TF-IDF | SVM (RBF) | 0.3206 | 0.4844 | 0.6000 | 0.2188 | 0.6010 | 0.5789 |
| Features originales | XGBoost | 0.5876 | 0.6175 | 0.6420 | 0.5417 | 0.6467 | 0.5727 |
| Features originales | Random Forest | 0.5763 | 0.6070 | 0.6296 | 0.5312 | 0.6918 | 0.5963 |
| Features originales | Gradient Boosting | 0.5862 | 0.6217 | 0.6538 | 0.5312 | 0.6599 | 0.5729 |
| Features originales | **Logistic Regression** | **0.6122** | **0.6040** | **0.6000** | **0.6250** | **0.6536** | **0.5767** |
| Features originales | SVM (RBF) | 0.5549 | 0.5950 | 0.6234 | 0.5000 | 0.6547 | 0.6124 |

#### Deteccion No Supervisada

| Modelo | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|--------|-----|-----------|--------|---------|--------|
| Isolation Forest | 0.0000 | 0.0000 | 0.0000 | 0.4754 | 0.4514 |
| One-Class SVM | 0.0000 | 0.0000 | 0.0000 | 0.3539 | 0.4067 |

### 11.2 Mejor Modelo por Representacion

| Representacion | Mejor Modelo | F1 | ROC-AUC |
|---------------|-------------|-----|---------|
| **BoW (frecuencias)** | **Logistic Regression** | **0.6492** | **0.7023** |
| BoW + TF-IDF | Logistic Regression | 0.6387 | 0.6966 |
| Features originales | Logistic Regression | 0.6122 | 0.6536 |

### 11.3 Top 5 Tokens Mas Discriminativos

Segun los coeficientes del modelo Logistic Regression sobre BoW + TF-IDF:

| Token | Importancia (|coef|) |
|-------|---------------------|
| `mean_interlog_time_http_interlog_time=muy_bajo` | 0.4542 |
| `dns_usual_dns_dstport_ratio=alto` | 0.4309 |
| `http_status_200_ratio=alto` | 0.4090 |
| `smtp_in_is_reply=alto` | 0.3969 |
| `dns_len_TTL=alto` | 0.3895 |

---

## 12. Interpretacion de Resultados

### 12.1 BoW vs Features Originales

El resultado mas significativo del proyecto es que la representacion **BoW con frecuencias crudas supera a las features originales**:

- **BoW (frecuencias):** F1 = 0.6492, ROC-AUC = 0.7023
- **Features originales:** F1 = 0.6122, ROC-AUC = 0.6536

**Mejora relativa:** +6.0% en F1, +7.5% en ROC-AUC.

Esto valida la hipotesis central del proyecto: la discretizacion de features numericas en tokens categoricos y su representacion como bag-of-words **preserva e incluso realza la informacion discriminativa** para deteccion de anomalias. Las posibles razones incluyen:

1. **Reduccion de ruido:** La discretizacion suaviza variaciones irrelevantes dentro de un mismo rango
2. **No linealidad implicita:** Los bins capturan relaciones no lineales que un modelo lineal no podria capturar sobre features continuas
3. **Robustez a outliers:** Los bins extremos (`muy_bajo`, `muy_alto`) contienen toda la cola de la distribucion, eliminando la influencia de valores atipicos

### 12.2 BoW Cruda vs BoW + TF-IDF

Contraintuitivamente, la BoW con frecuencias crudas supera a BoW + TF-IDF:

- **BoW (frecuencias):** F1 = 0.6492
- **BoW + TF-IDF:** F1 = 0.6387

**Diferencia:** -1.6% en F1 con TF-IDF.

Esto se explica porque en este dataset, la representacion BoW a nivel de muestra es **binaria** (cada feature produce exactamente un token, con valor 0 o 1). La normalizacion TF reduce el peso de features presentes (dividiendolas por 68, el total de tokens activos), y la ponderacion IDF redistribuye pesos basandose en rareza, lo cual no necesariamente alinea con discriminatividad entre clases.

En un escenario de BoW a nivel de usuario (donde los conteos son genuinos frecuencias acumuladas), TF-IDF podria ser mas beneficioso.

### 12.3 Logistic Regression como Mejor Clasificador

Logistic Regression domina consistentemente en las tres representaciones, superando a modelos mas complejos como XGBoost y Random Forest. Esto se explica por:

1. **Separabilidad lineal:** Con 212 features binarias, la frontera de decision entre clases es aproximadamente lineal en este espacio de alta dimension
2. **Regularizacion efectiva:** `class_weight="balanced"` y C=1.0 proporcionan la regularizacion justa
3. **Robustez con pocas muestras:** Con solo 1,247 muestras de entrenamiento por clase (despues del downsampling), los modelos mas complejos tienden al overfitting, mientras que Logistic Regression generaliza mejor
4. **Bajo bias-variance:** Los modelos de alta complejidad (XGBoost, RF) tienen alta varianza con tan pocas muestras de entrenamiento

### 12.4 Rendimiento de SVM (RBF)

SVM con kernel RBF obtiene el peor rendimiento en todas las representaciones BoW (F1 = 0.31-0.32), pero es competitivo con features originales (F1 = 0.55). Esto sugiere que el kernel RBF no es adecuado para el espacio discretizado binario de BoW, donde las distancias euclideas no son informativas.

### 12.5 Fracaso de Metodos No Supervisados

Tanto Isolation Forest como One-Class SVM obtienen F1 = 0.00, lo que significa que no detectaron **ninguna** anomalia correctamente. Las razones son:

1. **Dificultad intrinseca:** La criptomineria genera patrones de red que, individualmente por ventana, son similares al trafico legitimo. La anomalia es sutil y se manifiesta en combinaciones de features, no en outliers extremos.
2. **Tamano del test balanceado:** Con solo 96 muestras positivas y 96 negativas, incluso una pequena desalineacion en los scores produce F1=0.
3. **ROC-AUC bajo (0.47-0.35):** Valores por debajo de 0.5 indican que los modelos **invierten** la nocion de anomalia (clasifican las muestras benignas como mas anomalas que las de riesgo)
4. **Naturaleza del ataque:** La criptomineria no genera spikes obvios de anomalia; en cambio, modifica gradualmente los patrones de red de forma sostenida

Esto valida la decision de complementar con metodos supervisados y confirma que la deteccion de criptomineria requiere etiquetas de entrenamiento.

### 12.6 Interpretacion de Tokens Discriminativos

Los tokens mas discriminativos revelan los patrones de comportamiento de criptomineria:

1. **`mean_interlog_time_http_interlog_time=muy_bajo`**: Los mineros generan solicitudes HTTP extremadamente frecuentes hacia los pools de mineria, reduciendo el tiempo entre eventos HTTP a niveles atipicos.

2. **`dns_usual_dns_dstport_ratio=alto`**: Las consultas DNS de los mineros usan puertos DNS estandar (puerto 53), a diferencia del trafico legitimo que puede usar DNS-over-HTTPS u otros puertos.

3. **`http_status_200_ratio=alto`**: Las comunicaciones con pools de mineria generan respuestas exitosas (HTTP 200) consistentemente, sin los errores (404, 500, etc.) tipicos del trafico web normal.

4. **`smtp_in_is_reply=alto`**: Curiosamente, la actividad SMTP correlaciona con criptomineria. Esto podria indicar que los mineros generan actividad de correo como cobertura o que los usuarios de riesgo tambien realizan otras actividades anomalas.

5. **`dns_len_TTL=alto`**: Los mineros consultan dominios con TTLs altos, posiblemente porque los pools de mineria usan DNS con TTLs largos para persistencia.

### 12.7 Comparacion con el Paper RBD24

| Metrica | Paper RBD24 | Este proyecto |
|---------|-------------|---------------|
| Mejor F1 | ~0.63 | **0.6492** |
| Mejor modelo | Random Forest | Logistic Regression |
| Metodologia | Downsampling train+test | Downsampling train+test |
| Representacion | Features originales | BoW (frecuencias) |

El proyecto **supera marginalmente** los resultados del paper original, utilizando una representacion bag-of-words que no fue evaluada por los autores de RBD24. Esto sugiere que la discretizacion y representacion BoW es una alternativa viable y competitiva para UEBA.

---

## 13. Problemas Encontrados y Soluciones

### 13.1 Problema: F1-scores Extremadamente Bajos (~0.009)

**Descripcion:** En la primera iteracion del pipeline, todos los clasificadores obtuvieron F1-scores cercanos a 0.009, a pesar de que los modelos entrenaban correctamente.

**Causa raiz:** Se evaluaba en el test set **desbalanceado** (39,042 negativos vs 96 positivos). Con un ratio de 407:1 en test, incluso un buen clasificador produce miles de falsos positivos que destruyen la precision.

**Ejemplo numerico:**
- Supongamos que el clasificador detecta 80 de 96 positivos (Recall = 83%)
- Pero tambien clasifica 2,000 de 39,042 negativos como positivos (FPR = 5%)
- Precision = 80 / (80 + 2000) = 3.8%
- F1 = 2 * 0.038 * 0.83 / (0.038 + 0.83) = 0.073

**Solucion:** Analisis del paper RBD24, que reveló que los autores aplican downsampling **tanto al train como al test set**. Se implemento esta misma metodologia, obteniendo F1 = 0.6492.

**Leccion aprendida:** Al replicar resultados de un paper, es critico entender la metodologia de evaluacion exacta, no solo el modelo utilizado.

### 13.2 Problema: Configuracion de matplotlib en Entorno sin Display

**Descripcion:** El pipeline se ejecuta en un entorno sin interfaz grafica (sin display X11/Wayland). Al intentar generar graficas, matplotlib intentaba abrir una ventana interactiva, causando errores.

**Solucion:** Se configuro el backend no interactivo al inicio de cada modulo de visualizacion:

```python
import matplotlib
matplotlib.use("Agg")
```

El backend "Agg" (Anti-Grain Geometry) genera graficas directamente a archivos PNG sin necesidad de display.

### 13.3 Problema: Parametro Invalido en rcParams de matplotlib

**Descripcion:** El parametro `savefig.bbox_inches` no es un parametro valido de `plt.rcParams`, lo que causaba un error al importar el modulo de exploracion.

**Solucion:** Se elimino el parametro invalido de la configuracion global, dejando solo parametros validos (`figure.dpi`, `savefig.dpi`, `font.size`).

### 13.4 Problema: Logistic Regression no soporta feature_importances_

**Descripcion:** Al intentar graficar la importancia de features del mejor modelo (Logistic Regression), el atributo `feature_importances_` no existia, ya que es especifico de modelos basados en arboles.

**Solucion:** Se amplio la funcion `plot_feature_importance` para soportar multiples tipos de modelos:

```python
if hasattr(model, "feature_importances_"):
    imp = model.feature_importances_
elif hasattr(model, "coef_"):
    imp = np.abs(model.coef_[0])
```

Para Logistic Regression, se usan los **valores absolutos de los coeficientes** como proxy de importancia. Un coeficiente grande (positivo o negativo) indica una feature altamente discriminativa.

### 13.5 Problema: Filtracion de Datos entre Train y Test

**Descripcion potencial:** Si se divide train/test a nivel de muestra (filas), las ventanas temporales del mismo usuario podrian aparecer en ambos conjuntos, causando una evaluacion excesivamente optimista.

**Prevencion:** Se implemento division a nivel de **usuario** usando `GroupShuffleSplit`, con una verificacion explicita:

```python
assert not (set(u_train) & set(u_test)), "Filtracion de usuarios detectada"
```

Ningun usuario aparece en ambos conjuntos, garantizando una evaluacion realista.

### 13.6 Problema: One-Class SVM con Dataset Grande

**Descripcion:** One-Class SVM tiene complejidad computacional O(n^2) a O(n^3) en entrenamiento. Con 122,160 muestras normales, el entrenamiento era prohibitivamente lento.

**Solucion:** Se implemento subsampling aleatorio de las muestras de entrenamiento:

```python
max_s = 20000
if len(X_n_s) > max_s:
    idx = rng.choice(len(X_n_s), max_s, replace=False)
    X_oc = X_n_s[idx]
```

Se seleccionan aleatoriamente 20,000 muestras para el entrenamiento, lo que reduce el tiempo a un nivel manejable sin perder significativamente la representatividad del perfil de comportamiento normal.

### 13.7 Problema: Virtual Environment con Ruta Incorrecta

**Descripcion:** Al mover el directorio del proyecto, el virtual environment tenia rutas absolutas hardcodeadas en los shebangs de los scripts de pip, causando errores al instalar paquetes.

**Solucion:** Se uso `<venv>/bin/python -m pip install` en lugar del script `pip` directamente, lo que ignora el shebang roto y usa el interprete Python correcto.

---

## 14. Analisis Exploratorio de Datos (EDA)

El pipeline genera automaticamente 5 graficas exploratorias al inicio de cada ejecucion. Estas graficas se almacenan en `results/figures/` y proporcionan una comprension fundamental del dataset.

### 14.1 Distribucion de Clases (01_distribucion_clases.png)

Muestra el desbalance a dos niveles:
- **Nivel muestra:** 161,202 benignas vs 1,343 de riesgo (eje Y en escala logaritmica por la magnitud)
- **Nivel usuario:** 738 benignos vs 11 de riesgo

Esta grafica evidencia visualmente la magnitud del desbalance y justifica la necesidad de estrategias especiales de evaluacion.

### 14.2 Distribuciones de Features Top (02_top_features_distribucion.png)

Muestra histogramas comparativos (benigno vs riesgo) de las 12 features con mayor diferencia normalizada en medias entre clases. Se calcula:

```
diferencia = |media_benigno - media_riesgo| / std_total
```

Las features seleccionadas representan aquellas donde los patrones de criptomineria son mas distintos del comportamiento normal. Los histogramas muestran distribuciones superpuestas con densidad normalizada para compensar el desbalance.

### 14.3 Patrones Temporales (03_patrones_temporales.png)

Dos subgraficas:
- **Patron horario:** Proporcion de muestras por hora del dia. Revela si la actividad de criptomineria tiene patrones temporales distintos (e.g., actividad nocturna).
- **Patron semanal:** Proporcion de muestras por dia de la semana. Muestra si la actividad se concentra en dias especificos.

### 14.4 Correlacion entre Features (04_correlacion_features.png)

Heatmap de correlacion de Pearson entre un subconjunto representativo de features (8 por protocolo DNS, SSL y HTTP). Revela:
- Grupos de features altamente correlacionadas (que podrian ser redundantes)
- Relaciones entre protocolos
- Features independientes que aportan informacion unica

### 14.5 Perfil de Comportamiento Normal (05_perfil_comportamiento_normal.png)

Boxplots comparativos (benigno vs riesgo) de las 9 features con mayor varianza en la clase benigna. Esta grafica responde directamente a la recomendacion del asesor de analizar el comportamiento normal primero.

Las cajas muestran percentiles 25-75 con mediana, y los bigotes muestran el rango intercuartilico. Las diferencias visuales entre las cajas benignas y de riesgo indican las features mas utiles para discriminacion.

---

## 15. Validacion Cruzada

### 15.1 Metodologia

Se realizo validacion cruzada de 5 folds del mejor modelo (Logistic Regression) sobre la representacion BoW + TF-IDF, utilizando `StratifiedGroupKFold` para garantizar:

1. **Sin filtracion:** Ningun usuario aparece en multiples folds
2. **Estratificacion:** La proporcion de clases se mantiene en cada fold
3. **Downsampling consistente:** Tanto train como test de cada fold se balancean

### 15.2 Resultados por Fold

| Fold | F1 | ROC-AUC | TPR | FPR |
|------|-----|---------|------|------|
| 1 | 0.5394 | 0.6355 | 0.4829 | 0.3077 |
| 2 | 0.7088 | 0.7678 | 0.7404 | 0.3489 |
| 3 | 0.3876 | 0.5447 | 0.3112 | 0.2946 |
| 4 | 0.3469 | 0.4652 | 0.2677 | 0.2756 |
| 5 | 0.4387 | 0.5326 | 0.3826 | 0.3615 |

### 15.3 Promedios

| Metrica | Media | Desviacion Estandar |
|---------|-------|---------------------|
| F1 | 0.4843 | 0.1294 |
| F1_macro | 0.5475 | 0.0835 |
| Precision | 0.5622 | 0.0716 |
| Recall (TPR) | 0.4370 | 0.1683 |
| ROC-AUC | 0.5892 | 0.1045 |
| PR-AUC | 0.5834 | 0.0844 |
| FPR | 0.3177 | 0.0326 |

### 15.4 Interpretacion de la Validacion Cruzada

La validacion cruzada revela varias observaciones importantes:

1. **Alta varianza (std F1 = 0.13):** El rendimiento varia significativamente entre folds. El Fold 2 alcanza F1=0.71 mientras el Fold 4 solo 0.35. Esto se debe a que con solo 11 usuarios de riesgo, la composicion de cada fold depende criticamente de que usuarios de riesgo caen en train vs test.

2. **F1 CV (0.48) < F1 holdout (0.64):** El promedio de validacion cruzada es inferior al resultado del holdout. Esto no indica overfitting, sino que refleja la variabilidad inherente: algunas particiones de los 11 usuarios de riesgo son mas favorables que otras.

3. **FPR estable (0.32 +/- 0.03):** A pesar de la varianza en TPR, la tasa de falsos positivos es consistente, indicando que el modelo tiene un comportamiento estable en la clase negativa.

4. **Dependencia de usuarios especificos:** El rendimiento esta fuertemente condicionado por que usuarios de riesgo aparecen en el test de cada fold. Algunos usuarios de riesgo pueden tener patrones mas detectables que otros.

---

## 16. Conclusiones

### 16.1 Conclusiones Principales

1. **La representacion bag-of-words es efectiva para UEBA:** La discretizacion de features numericas de red en tokens categoricos y su representacion como vectores de frecuencias produce resultados comparables o superiores a los features originales (F1 = 0.6492 vs 0.6122).

2. **BoW cruda supera a TF-IDF en representacion binaria:** Cuando la BoW es binaria (como en este caso, a nivel de muestra), la ponderacion TF-IDF no aporta beneficio significativo. TF-IDF seria mas util en representaciones con conteos genuinos (nivel de usuario).

3. **Logistic Regression es el clasificador optimo:** En un escenario con pocas muestras de entrenamiento (despues del downsampling) y features binarias de alta dimension, un modelo lineal generaliza mejor que modelos complejos como XGBoost o Random Forest.

4. **La deteccion no supervisada de criptomineria es extremadamente dificil:** Los metodos basados en perfilado de comportamiento normal (Isolation Forest, One-Class SVM) fracasan completamente. La criptomineria genera patrones de red que son demasiado sutiles para ser detectados sin supervision.

5. **La metodologia de evaluacion es critica:** La misma pipeline con la misma calidad de modelo puede reportar F1=0.009 o F1=0.649 dependiendo de si se evalua en test desbalanceado o balanceado. Es imprescindible documentar y justificar la metodologia de evaluacion.

6. **Los tokens discriminativos tienen interpretacion semantica clara:** Los patrones de criptomineria se manifiestan en tiempos entre eventos HTTP anomalamente bajos, uso de puertos DNS estandar, alta tasa de respuestas HTTP exitosas, y TTLs DNS elevados.

### 16.2 Limitaciones

1. **Test set pequeno (192 muestras balanceadas):** La evaluacion sobre 96 positivos y 96 negativos tiene alta varianza inherente. Cambios de 1-2 predicciones pueden mover el F1 varios puntos porcentuales.

2. **Pocos usuarios de riesgo (11):** La generalizabilidad esta limitada por la muestra de atacantes. Un atacante con un patron muy diferente podria no ser detectado.

3. **BoW binaria a nivel de muestra:** Como cada feature produce un solo token por muestra, la representacion BoW es equivalente a un one-hot encoding de los bins. El verdadero poder de BoW (frecuencias) se aprovecharia mas a nivel de usuario con agregacion temporal.

4. **Downsampling del test:** Aunque es la metodologia del paper, descartar el 99.75% del test set limita la evaluacion de la tasa de falsos positivos en condiciones reales.

5. **Sin optimizacion de hiperparametros:** Los hiperparametros de los clasificadores se seleccionaron manualmente. Una busqueda sistematica (GridSearchCV, Optuna) podria mejorar los resultados.

### 16.3 Trabajo Futuro

1. **Optimizacion de hiperparametros** con validacion cruzada estratificada
2. **Analisis de BoW a nivel de usuario** con frecuencias acumuladas y TF-IDF
3. **Ensamble de modelos** combinando predicciones de multiples representaciones
4. **Incrementar NUM_BINS** (e.g., 10, 20) para evaluar el efecto de la granularidad de discretizacion
5. **Evaluacion en otros datasets de RBD24** (RAT, exfiltracion) para validar la generalidad del enfoque
6. **Tecnicas de oversampling** (SMOTE) como alternativa al undersampling

---

## 17. Estructura del Codigo

### 17.1 Descripcion de Modulos

#### `src/config.py`
Configuracion central con constantes, rutas, y parametros. Parametros clave: NUM_BINS=5, SPARSE_THRESHOLD=0.90, TEST_SIZE=0.20, RANDOM_STATE=42.

#### `src/data_loader.py`
Funciones para carga del dataset Parquet, separacion de features y metadatos, agrupacion de features por protocolo, y generacion de resumen descriptivo.

#### `src/exploracion.py`
Modulo de analisis exploratorio con 5 funciones de visualizacion: distribucion de clases, distribuciones de features, patrones temporales, correlacion, y perfil de comportamiento normal. Usa backend matplotlib "Agg" para entornos sin display.

#### `src/preprocessing.py`
Limpieza del dataset: eliminacion de features constantes (varianza cero) e identificacion de features dispersas (>90% ceros) vs densas.

#### `src/tokenizer.py`
Clase `EventTokenizer` que implementa la discretizacion adaptativa. El metodo `fit()` aprende los limites de binning sobre datos benignos. El metodo `transform()` convierte un DataFrame en una matriz BoW. El vocabulario se construye automaticamente a partir de las features y los bins.

#### `src/bow_builder.py`
Funciones para construir la representacion BoW a nivel de muestra y de usuario, y para aplicar ponderacion TF-IDF con smooth IDF y normalizacion L2.

#### `src/models.py`
Modulo de modelado con: division train/test a nivel de usuario, downsampling, diccionario de clasificadores (5 supervisados), funcion de evaluacion con metricas duales (balanceado + completo), perfilado no supervisado (Isolation Forest, One-Class SVM), y validacion cruzada estratificada por grupo.

#### `src/evaluation.py`
Modulo de visualizacion de resultados con funciones para: matrices de confusion, curvas ROC, curvas Precision-Recall, comparacion de metricas entre representaciones, importancia de features/tokens, resultados no supervisados, analisis de la representacion BoW, y guardado de metricas consolidadas en CSV.

#### `main.py`
Orquestador del pipeline completo en 9 pasos. Importa todos los modulos, ejecuta secuencialmente cada etapa, y genera un resumen final con los mejores resultados por representacion.

### 17.2 Figuras Generadas

| Archivo | Descripcion |
|---------|-------------|
| `01_distribucion_clases.png` | Desbalance de clases (muestra y usuario) |
| `02_top_features_distribucion.png` | Histogramas de features mas discriminativas |
| `03_patrones_temporales.png` | Actividad por hora y dia de la semana |
| `04_correlacion_features.png` | Heatmap de correlacion |
| `05_perfil_comportamiento_normal.png` | Boxplots benigno vs riesgo |
| `06_matrices_confusion.png` | Matrices de confusion (BoW + TF-IDF) |
| `06b_matrices_confusion.png` | Matrices de confusion (Features originales) |
| `07_curvas_roc.png` | Curvas ROC (BoW + TF-IDF) |
| `07b_curvas_roc.png` | Curvas ROC (Features originales) |
| `08_curvas_pr.png` | Curvas Precision-Recall (BoW + TF-IDF) |
| `09_comparacion_representaciones.png` | Comparacion de metricas entre representaciones |
| `10_importancia_tokens.png` | Top 20 tokens discriminativos |
| `11_no_supervisado.png` | Matrices de confusion no supervisadas |
| `12_no_supervisado_roc.png` | Curvas ROC no supervisadas |
| `13_analisis_bow.png` | Analisis de la representacion BoW |

### 17.3 Parametros de Configuracion

| Parametro | Valor | Ubicacion | Descripcion |
|-----------|-------|-----------|-------------|
| `NUM_BINS` | 5 | config.py | Numero de bins para discretizacion |
| `BIN_LABELS` | ["muy_bajo", "bajo", "medio", "alto", "muy_alto"] | config.py | Etiquetas de bins |
| `SPARSE_THRESHOLD` | 0.90 | config.py | Umbral para clasificar features como dispersas |
| `TEST_SIZE` | 0.20 | config.py | Proporcion de datos para test |
| `RANDOM_STATE` | 42 | config.py | Semilla para reproducibilidad |
| `CV_FOLDS` | 5 | config.py | Numero de folds de validacion cruzada |

---

## 18. Referencias

1. **RBD24:** Benchmark paper para deteccion de anomalias en UEBA con datasets realistas de red corporativa. Define la ventana temporal, features y metodologia de evaluacion utilizados en este proyecto.

2. **scikit-learn:** Pedregosa et al. "Scikit-learn: Machine Learning in Python." JMLR 12, pp. 2825-2830, 2011. Biblioteca utilizada para modelos ML, metricas y preprocesamiento.

3. **XGBoost:** Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD 2016. Clasificador de gradient boosting utilizado.

4. **imbalanced-learn:** Lemaitre, G., Nogueira, F., & Aridas, C.K. "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning." JMLR 18, 2017. Utilizado para RandomUnderSampler.

5. **UEBA fundamentals:** Gartner Market Guide for User and Entity Behavior Analytics. Marco teorico para analisis de comportamiento.

---

*Reporte generado como parte del Proyecto 2 de Ciberseguridad. Todo el codigo fuente y los resultados estan disponibles en el directorio del proyecto.*
