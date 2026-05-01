"""
Configuracion central del proyecto UEBA - Extraccion de caracteristicas bag-of-words.
Dataset: RBD24 - Crypto Desktop (deteccion de actividad de criptomineria).
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

DATASET_PATH = DATA_DIR / "Crypto_desktop.parquet"

for d in [RESULTS_DIR, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Columnas de metadatos (no son features numericas)
# ---------------------------------------------------------------------------
META_COLS = ["entity", "label", "user_id", "timestamp"]
LABEL_COL = "label"
USER_COL = "user_id"
ENTITY_COL = "entity"
TIMESTAMP_COL = "timestamp"

# ---------------------------------------------------------------------------
# Grupos de features por protocolo (para tokenizacion semantica)
# ---------------------------------------------------------------------------
DNS_FEATURES_PREFIX = "dns_"
SSL_FEATURES_PREFIX = "ssl_"
HTTP_FEATURES_PREFIX = "http_"
SMTP_FEATURES_PREFIX = "smtp_"
TEMPORAL_FEATURES_PREFIX = "non_working_"

# ---------------------------------------------------------------------------
# Parametros de tokenizacion
# ---------------------------------------------------------------------------
NUM_BINS = 5
BIN_LABELS = ["muy_bajo", "bajo", "medio", "alto", "muy_alto"]
SPARSE_THRESHOLD = 0.90  # si >90% de valores son 0, tratamiento especial

# ---------------------------------------------------------------------------
# Parametros de modelado
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5
