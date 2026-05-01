# UEBA con Bag-of-Words — Detección de Amenazas en Tráfico de Red

> **User and Entity Behavior Analytics (UEBA)** usando *frequency counting* (bag-of-words) sobre features de tráfico de red para detección de amenazas de ciberseguridad.

---

## Descripción General

Este proyecto implementa un pipeline completo de UEBA que discretiza features numéricas de tráfico de red en tokens categóricos y aplica representaciones **bag-of-words (BoW)** para entrenar clasificadores supervisados de detección de amenazas.

Se evalúan **dos escenarios de amenaza** independientes:

| Escenario | Dataset | Amenaza detectada | Pipeline |
|-----------|---------|-------------------|---------|
| **Crypto** | RBD24 – Crypto Desktop | Actividad de criptominería en workstations | `main.py` |
| **P2P** | ISCXVPN2016 – Scenario B | Accesos no autorizados vía redes P2P / P2P-over-VPN | `main_p2p.py` |

---

## Arquitectura del Sistema

```
Dataset (Parquet/ARFF)
        │
        ▼
┌──────────────────┐
│  data_loader.py  │  ← Carga, validación y resumen del dataset
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  exploracion.py  │  ← EDA: distribuciones, correlaciones, patrones temporales
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  preprocessing.py    │  ← Eliminación de features constantes/dispersas
└────────┬─────────────┘
         │
         ▼
┌──────────────────┐
│  tokenizer.py    │  ← Discretización adaptativa → tokens "{feat}={bin}"
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  bow_builder.py  │  ← Matrices BoW (muestra y usuario) + TF-IDF
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   models.py      │  ← Clasificadores supervisados + validación cruzada
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  evaluation.py   │  ← Métricas, curvas ROC/PR, matrices de confusión
└──────────────────┘
```

### Representaciones evaluadas

| Representación | Descripción |
|----------------|-------------|
| **BoW (frecuencias)** | Conteo directo de tokens por ventana temporal |
| **BoW + TF-IDF** | BoW ponderado por relevancia del token (penaliza tokens ubicuos) |
| **Features originales** | Baseline: features numéricas sin discretizar |

### Clasificadores

- XGBoost
- Random Forest
- Gradient Boosting
- Logistic Regression
- SVM (kernel RBF)

---

## Estructura del Repositorio

```
Proyecto_Prog/
│
├── main.py                    # Pipeline principal — Dataset Crypto (RBD24)
├── main_p2p.py                # Pipeline P2P — Dataset ISCXVPN2016
├── prepare_p2p_dataset.py     # Adaptador ARFF → Parquet para el pipeline P2P
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Parámetros globales y rutas
│   ├── data_loader.py         # Carga y validación del dataset
│   ├── exploracion.py         # Análisis exploratorio (EDA) y visualizaciones
│   ├── preprocessing.py       # Limpieza y selección de features
│   ├── tokenizer.py           # Discretización adaptativa → vocabulario BoW
│   ├── bow_builder.py         # Construcción de matrices BoW y TF-IDF
│   ├── models.py              # Entrenamiento y evaluación de clasificadores
│   └── evaluation.py         # Métricas, gráficas y exportación de resultados
│
├── Scenario B-ARFF/           # Archivos ARFF del dataset ISCXVPN2016
│   ├── TimeBasedFeatures-Dataset-15s.arff
│   ├── TimeBasedFeatures-Dataset-30s.arff
│   ├── TimeBasedFeatures-Dataset-60s.arff
│   └── TimeBasedFeatures-Dataset-120s.arff
│
├── results/
│   ├── figures/               # Gráficas del pipeline Crypto
│   ├── metrics/
│   │   └── resultados_consolidados.csv
│   └── p2p/
│       ├── figures/           # Gráficas del pipeline P2P
│       └── metrics/
│           └── resultados_consolidados.csv
│
├── REPORTE_PROYECTO.md        # Reporte técnico detallado del proyecto
├── requirements.txt
└── .gitignore
```

---

## Instalación

### Requisitos previos

- Python 3.10+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/danielportela007/ueba-bow-network-threats.git
cd ueba-bow-network-threats

# 2. Crear y activar el entorno virtual
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### Escenario 1 — Detección de Criptominería (RBD24)

> Requiere el dataset `Crypto_desktop.parquet` en el directorio raíz.

```bash
python main.py
```

### Escenario 2 — Detección de Tráfico P2P No Autorizado (ISCXVPN2016)

```bash
# Paso 1: Preparar el dataset desde los archivos ARFF
python prepare_p2p_dataset.py

# Paso 2: Ejecutar el pipeline P2P
python main_p2p.py
```

---

## Resultados

### Dataset Crypto (RBD24)

| Representación | Mejor modelo | F1 | ROC-AUC |
|----------------|--------------|-----|---------|
| BoW (frecuencias) | Logistic Regression | 0.649 | 0.702 |
| BoW + TF-IDF | Logistic Regression | 0.639 | 0.697 |
| Features originales | Logistic Regression | 0.612 | 0.654 |

### Dataset P2P (ISCXVPN2016)

| Representación | Mejor modelo | F1 | ROC-AUC |
|----------------|--------------|-----|---------|
| BoW (frecuencias) | SVM (RBF) | 0.935 | 0.955 |
| BoW + TF-IDF | SVM (RBF) | 0.934 | 0.956 |
| Features originales | Random Forest | 0.936 | 0.973 |

Los resultados completos (todas las representaciones × clasificadores × métricas) se exportan automáticamente en:
- `results/metrics/resultados_consolidados.csv`
- `results/p2p/metrics/resultados_consolidados.csv`

---

## Metodología

### Tokenización adaptativa

Cada feature numérica se discretiza en bins y se convierte en un token con el formato `{feature_name}={bin_label}`:

- **Features densas** → binning por cuantiles (e.g., `flowBytesPerSecond=alto`)
- **Features dispersas** (>90 % ceros) → categoría `cero` + bins para valores no-nulos (e.g., `dns_len_TTL=cero`)

El tokenizador se ajusta **exclusivamente sobre datos benignos** para capturar el perfil de comportamiento normal.

### Manejo del desbalance de clases

Se aplica **downsampling** de la clase mayoritaria en el conjunto de entrenamiento (estrategia adoptada del benchmark RBD24), combinado con `class_weight="balanced"` en los clasificadores que lo admiten.

### División train/test

La separación se realiza a **nivel de usuario** (`GroupShuffleSplit`) para garantizar que no haya filtración de datos entre conjuntos.

---

## Módulos — Descripción Técnica

| Módulo | Responsabilidad |
|--------|----------------|
| `config.py` | Rutas, constantes y hiperparámetros globales |
| `data_loader.py` | Lectura del `.parquet`, separación features/metadatos, resumen estadístico |
| `exploracion.py` | EDA: distribución de clases, histogramas comparativos, patrones temporales |
| `preprocessing.py` | Eliminación de features constantes, identificación de features dispersas |
| `tokenizer.py` | `EventTokenizer`: fit/transform para discretización adaptativa |
| `bow_builder.py` | Matrices BoW a nivel muestra y usuario, normalización TF-IDF |
| `models.py` | División train/test, downsampling, entrenamiento y métricas de clasificadores |
| `evaluation.py` | Matrices de confusión, curvas ROC/PR, comparación de representaciones, importancia de tokens |

---

## Referencia del Dataset

- **RBD24:** Realistic Baseline for anomaly Detection 2024 — benchmark UEBA para entornos corporativos.
- **ISCXVPN2016:** Canadian Institute for Cybersecurity — Scenario B (Time-Based Features). [Enlace oficial](https://www.unb.ca/cic/datasets/vpn.html)

---

## Autor

**Daniel Portela**  
Ciberseguridad — Proyecto 2  
Abril 2026
