"""
Adaptador del dataset ISCXVPN2016 - Scenario B para el pipeline UEBA.

Dataset original:
  ISCXVPN2016 (Canadian Institute for Cybersecurity)
  Scenario B - Time-Based Features (sin VPN)
  Archivos usados:
    TimeBasedFeatures-Dataset-15s.arff
    TimeBasedFeatures-Dataset-30s.arff
    TimeBasedFeatures-Dataset-60s.arff
    TimeBasedFeatures-Dataset-120s.arff

Cada fila es una ventana temporal de trafico de red clasificada por aplicacion:
  BROWSING, CHAT, STREAMING, MAIL, VOIP, FT  -> label=0 (trafico corporativo normal)
  P2P                                         -> label=1 (acceso P2P no autorizado)

Transformaciones para compatibilidad con el pipeline UEBA existente:
  1. Combina los 4 archivos ARFF (diferentes ventanas temporales).
  2. Limpia valores -1 (indicadores de flujo unidireccional sin IAT inverso).
  3. Crea columnas entity, user_id, timestamp sinteticas para simular el
     contexto de workstations corporativas con captura por ventana horaria.
     - 50 usuarios benignos: solo trafico corporativo.
     - 10 usuarios P2P: mezcla de trafico corporativo + P2P (60% P2P / 40% normal).
  4. Guarda como P2P_ISCXVPN.parquet en el directorio raiz del proyecto.

Uso:
  python prepare_p2p_dataset.py

Salida:
  P2P_ISCXVPN.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
SEED = 42
ARFF_DIR = Path(__file__).parent / "Scenario B-ARFF"
OUTPUT_PATH = Path(__file__).parent / "P2P_ISCXVPN.parquet"
START_DATE = pd.Timestamp("2024-03-04 00:00:00")

N_BENIGN_USERS = 50
N_P2P_USERS = 10
P2P_WINDOW_RATIO = 0.60   # fraccion de ventanas P2P para usuarios maliciosos

ARFF_FILES = [
    "TimeBasedFeatures-Dataset-15s.arff",
    "TimeBasedFeatures-Dataset-30s.arff",
    "TimeBasedFeatures-Dataset-60s.arff",
    "TimeBasedFeatures-Dataset-120s.arff",
]

# Clases que se consideran P2P no autorizado (incluyendo P2P sobre VPN)
P2P_CLASSES = {"P2P", "VPN-P2P"}

# Nombres de columna en el orden en que aparecen en los ARFF
FEATURE_COLS = [
    "duration", "total_fiat", "total_biat",
    "min_fiat", "min_biat", "max_fiat", "max_biat",
    "mean_fiat", "mean_biat",
    "flowPktsPerSecond", "flowBytesPerSecond",
    "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle",
]

rng = np.random.RandomState(SEED)


# ---------------------------------------------------------------------------
# Carga y parseo de ARFF
# ---------------------------------------------------------------------------

def _parse_arff(path: Path) -> pd.DataFrame:
    """
    Parsea la seccion DATA de un archivo ARFF como CSV.
    Los archivos ISCXVPN2016 tienen exactamente 23 features numericas + 1 clase.
    Los valores -1 son indicadores de flujo unidireccional (sin IAT inverso).
    """
    rows = []
    in_data = False
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("@DATA"):
                in_data = True
                continue
            if in_data and line and not line.startswith("@"):
                # Separar en partes: los datos tienen exactamente 24 campos
                parts = line.split(",")
                if len(parts) < 24:
                    continue
                # Ultimos campos extras (artefacto del formato del archivo)
                numeric_part = parts[:23]
                class_part = parts[23].strip()
                rows.append(numeric_part + [class_part])

    cols = FEATURE_COLS + ["traffic_class"]
    df = pd.DataFrame(rows, columns=cols)

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_all_arff() -> pd.DataFrame:
    """Carga y combina los 4 archivos ARFF."""
    frames = []
    for fname in ARFF_FILES:
        path = ARFF_DIR / fname
        if not path.exists():
            print(f"  [ADVERTENCIA] No encontrado: {path}")
            continue
        df_file = _parse_arff(path)
        # "TimeBasedFeatures-Dataset-15s.arff" -> split by "-" gives index 3 = "15s.arff"
        window_label = fname.replace(".arff", "").split("-")[-1]   # "15s", "30s", etc.
        df_file["time_window"] = window_label
        frames.append(df_file)
        n_p2p_file = df_file["traffic_class"].isin(P2P_CLASSES).sum()
        print(f"  Cargado {fname}: {len(df_file):,} filas "
              f"(P2P/VPN-P2P={n_p2p_file:,})")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Limpieza de features
# ---------------------------------------------------------------------------

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tratamiento de valores especiales:
    -1  aparece en columnas de IAT inverso cuando el flujo es unidireccional
        (solo paquetes forward, sin respuesta). Se reemplaza con 0 ya que
        la ausencia de trafico bidireccional es informacion relevante.
    NaN se rellena con 0 de forma conservadora.
    """
    for col in FEATURE_COLS:
        # Sustituir -1 por 0 (flujo unidireccional = sin trafico de retorno)
        df[col] = df[col].where(df[col] >= 0, 0.0)
        df[col] = df[col].fillna(0.0).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Asignacion de usuarios y timestamps
# ---------------------------------------------------------------------------

def assign_users_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columnas entity, user_id, label y timestamp compatibles con el
    pipeline UEBA existente.

    Logica de asignacion:
    - Usuarios P2P: mezcla de ventanas P2P + ventanas benignas (60/40).
      Refleja el comportamiento real: un empleado que usa P2P sigue
      haciendo navegacion normal y email el resto del tiempo.
    - Usuarios benignos: solo ventanas de trafico corporativo.

    Timestamps: secuenciales por usuario a partir de START_DATE con
    salto de 10 minutos (mismo stride que el dataset Crypto Desktop).
    """
    p2p_mask = df["traffic_class"].isin(P2P_CLASSES)
    benign_mask = ~p2p_mask

    p2p_idx = df.index[p2p_mask].tolist()
    benign_idx = df.index[benign_mask].tolist()

    rng.shuffle(p2p_idx)
    rng.shuffle(benign_idx)

    # -- Usuarios P2P --
    # Cada usuario P2P necesita ~P2P_WINDOW_RATIO de ventanas P2P
    # => si N_P2P ventanas P2P por usuario, necesita N_P2P*(1-r)/r ventanas benignas
    n_p2p_total = len(p2p_idx)
    p2p_per_user = n_p2p_total // N_P2P_USERS

    # Ventanas benignas que se asignan a usuarios P2P
    benign_for_p2p = int(p2p_per_user * (1 - P2P_WINDOW_RATIO) / P2P_WINDOW_RATIO)
    benign_for_p2p_total = benign_for_p2p * N_P2P_USERS
    benign_for_benign_users = benign_idx[benign_for_p2p_total:]
    benign_for_p2p_idx = benign_idx[:benign_for_p2p_total]

    # -- Usuarios benignos --
    benign_per_user = len(benign_for_benign_users) // N_BENIGN_USERS

    # Construir asignaciones: lista de (user_id, [indices])
    assignments = {}

    for u in range(N_P2P_USERS):
        uid = f"U-P2P-{u+1:02d}"
        ent = f"WS-P2P-{u+1:02d}"
        p2p_slice = p2p_idx[u * p2p_per_user: (u+1) * p2p_per_user]
        benign_slice = benign_for_p2p_idx[u * benign_for_p2p: (u+1) * benign_for_p2p]
        assignments[uid] = {"entity": ent, "indices": p2p_slice + benign_slice}

    for u in range(N_BENIGN_USERS):
        uid = f"U-BEN-{u+1:02d}"
        ent = f"WS-BEN-{u+1:02d}"
        benign_slice = benign_for_benign_users[
            u * benign_per_user: (u+1) * benign_per_user
        ]
        assignments[uid] = {"entity": ent, "indices": benign_slice}

    # Construir columnas entity, user_id, label, timestamp
    entity_col = pd.Series(index=df.index, dtype=str)
    userid_col = pd.Series(index=df.index, dtype=str)
    label_col = pd.Series(index=df.index, dtype=np.int32)
    timestamp_col = pd.Series(index=df.index, dtype="datetime64[ns]")

    for uid, info in assignments.items():
        idxs = info["indices"]
        if not idxs:
            continue
        # Shuffle to mix P2P and benign windows within each user
        rng.shuffle(idxs)
        entity_col.loc[idxs] = info["entity"]
        userid_col.loc[idxs] = uid
        label_col.loc[idxs] = df.loc[idxs, "traffic_class"].apply(
            lambda c: 1 if c in P2P_CLASSES else 0
        ).values
        # Timestamps secuenciales con stride de 10 minutos
        ts = pd.date_range(START_DATE, periods=len(idxs), freq="10min")
        timestamp_col.loc[idxs] = ts.values

    df["entity"] = entity_col
    df["user_id"] = userid_col
    df["label"] = label_col
    df["timestamp"] = timestamp_col

    # Eliminar filas sin usuario asignado (sobrantes)
    df = df.dropna(subset=["user_id"]).copy()
    df["label"] = df["label"].astype(np.int32)

    return df


# ---------------------------------------------------------------------------
# Resumen del dataset final
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame) -> None:
    feat_cols = FEATURE_COLS
    vc = df["label"].value_counts().sort_index()

    print("=" * 65)
    print("DATASET REAL: ISCXVPN2016 - Accesos P2P no autorizados")
    print("Fuente: Canadian Institute for Cybersecurity - Scenario B")
    print("=" * 65)
    print(f"Muestras totales:     {len(df):,}")
    print(f"Features numericas:   {len(feat_cols)}")
    print(f"Usuarios unicos:      {df['user_id'].nunique()}")
    print(f"Rango temporal:       {df['timestamp'].min()} - "
          f"{df['timestamp'].max()}")
    print()
    print("Distribucion de clases:")
    for lbl, count in vc.items():
        pct = 100 * count / len(df)
        tag = "Benigno (corp.) " if lbl == 0 else "P2P no autorizado"
        print(f"  {tag} (label={lbl}): {count:>7,} ({pct:.2f}%)")
    ratio = vc[0] / vc[1]
    print(f"\nRatio de desbalance:  {ratio:.1f}:1")
    print()

    user_lbl = df.groupby("user_id")["label"].max()
    n_ben = (user_lbl == 0).sum()
    n_mal = (user_lbl == 1).sum()
    print(f"Usuarios benignos:    {n_ben}")
    print(f"Usuarios P2P:         {n_mal}")
    print()

    print("Distribucion de clases de trafico originales:")
    tc_vc = df["traffic_class"].value_counts()
    for tc, cnt in tc_vc.items():
        print(f"  {tc:<12}: {cnt:>6,}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    print("Procesando dataset ISCXVPN2016 - Scenario B...")
    print()

    # 1. Cargar ARFF
    print("[1/4] Cargando archivos ARFF...")
    df = load_all_arff()
    # Eliminar filas con clase vacia o desconocida
    df = df[df["traffic_class"].str.strip() != ""].copy()
    n_p2p = df["traffic_class"].isin(P2P_CLASSES).sum()
    print(f"  Total combinado: {len(df):,} filas (P2P+VPN-P2P={n_p2p:,})")

    # 2. Limpiar features
    print("\n[2/4] Limpiando features...")
    df = clean_features(df)
    neg_before = (df[FEATURE_COLS] < 0).sum().sum()
    print(f"  Valores negativos restantes: {neg_before}")

    # 3. Asignar usuarios y timestamps
    print("\n[3/4] Asignando usuarios y timestamps...")
    df = assign_users_timestamps(df)

    # 4. Guardar
    print("\n[4/4] Guardando parquet...")
    meta_cols = ["entity", "label", "user_id", "timestamp"]
    extra_cols = ["traffic_class", "time_window"]
    feat_order = sorted(FEATURE_COLS)
    final_cols = meta_cols + feat_order + [c for c in extra_cols if c in df.columns]
    df = df[final_cols]

    # Features a float32
    for col in FEATURE_COLS:
        df[col] = df[col].astype(np.float32)

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print()
    dataset_summary(df)
    print(f"Guardado en: {OUTPUT_PATH}")
    print(f"Tamano: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
