"""
Carga y validacion inicial del dataset RBD24 Crypto Desktop.
"""
import pandas as pd
import numpy as np
from . import config


def load_dataset(path=None):
    """Carga el archivo parquet y retorna el DataFrame."""
    path = path or config.DATASET_PATH
    df = pd.read_parquet(path)
    return df


def split_features_metadata(df):
    """Separa las columnas de features numericas de las de metadatos.

    Returns
    -------
    feature_cols : list[str]
        Columnas numericas utilizables como features.
    meta_cols : list[str]
        Columnas de metadatos (label, user_id, entity, timestamp).
    """
    meta_cols = [c for c in config.META_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return feature_cols, meta_cols


def get_feature_groups(feature_cols):
    """Agrupa features por protocolo para analisis semantico."""
    groups = {
        "dns": [],
        "ssl": [],
        "http": [],
        "smtp": [],
        "temporal": [],
        "interlog": [],
    }
    for col in feature_cols:
        if "interlog_time" in col:
            groups["interlog"].append(col)
        elif col.startswith("dns_"):
            groups["dns"].append(col)
        elif col.startswith("ssl_"):
            groups["ssl"].append(col)
        elif col.startswith("http_"):
            groups["http"].append(col)
        elif col.startswith("smtp_"):
            groups["smtp"].append(col)
        elif col.startswith("non_working_"):
            groups["temporal"].append(col)
    return groups


def dataset_summary(df, feature_cols):
    """Imprime un resumen del dataset."""
    print("=" * 60)
    print("RESUMEN DEL DATASET - RBD24 Crypto Desktop")
    print("=" * 60)
    print(f"Muestras totales:       {len(df):,}")
    print(f"Features numericas:     {len(feature_cols)}")
    print(f"Usuarios unicos:        {df[config.USER_COL].nunique()}")
    print(f"Rango temporal:         {df[config.TIMESTAMP_COL].min()} - "
          f"{df[config.TIMESTAMP_COL].max()}")
    print()
    print("Distribucion de clases:")
    vc = df[config.LABEL_COL].value_counts()
    for label, count in vc.items():
        pct = 100 * count / len(df)
        tag = "Benigno" if label == 0 else "Riesgo (Crypto)"
        print(f"  {tag} (label={label}): {count:>7,} ({pct:.2f}%)")
    print()
    ratio = vc[0] / vc[1]
    print(f"Ratio de desbalance:    {ratio:.1f}:1")
    print()
    print("Usuarios por clase:")
    for label in sorted(df[config.LABEL_COL].unique()):
        n = df[df[config.LABEL_COL] == label][config.USER_COL].nunique()
        tag = "Benigno" if label == 0 else "Riesgo"
        print(f"  {tag}: {n} usuarios")
    print("=" * 60)



if __name__ == "__main__":
    df = load_dataset()
    feature_cols, _ = split_features_metadata(df)
    dataset_summary(df, feature_cols)