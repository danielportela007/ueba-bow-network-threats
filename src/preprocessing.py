
"""
Preprocesamiento del dataset.
- Eliminacion de features constantes
- Filtrado de features con varianza cero
- Normalizacion opcional
"""
import numpy as np
import pandas as pd
from . import config


def remove_constant_features(df, feature_cols):
    """Elimina features con varianza cero (constantes en todo el dataset).

    Returns
    -------
    kept : list[str]
        Features retenidas.
    removed : list[str]
        Features eliminadas.
    """
    removed = []
    kept = []
    for col in feature_cols:
        if df[col].std() == 0:
            removed.append(col)
        else:
            kept.append(col)
    return kept, removed


def identify_sparse_features(df, feature_cols, threshold=None):
    """Identifica features con alto porcentaje de valores en cero.

    Parameters
    ----------
    threshold : float
        Proporcion minima de ceros para considerar la feature como dispersa.

    Returns
    -------
    sparse_cols : list[str]
        Features con >threshold proporcion de ceros.
    dense_cols : list[str]
        Features no dispersas.
    """
    if threshold is None:
        threshold = config.SPARSE_THRESHOLD
    sparse_cols = []
    dense_cols = []
    for col in feature_cols:
        zero_ratio = (df[col] == 0).mean()
        if zero_ratio >= threshold:
            sparse_cols.append(col)
        else:
            dense_cols.append(col)
    return sparse_cols, dense_cols


def preprocess(df, feature_cols):
    """Pipeline de preprocesamiento completo.

    Returns
    -------
    df : pd.DataFrame
        DataFrame sin cambios (se modifican solo las listas de features).
    clean_features : list[str]
        Features limpias para tokenizacion.
    sparse_features : list[str]
        Features dispersas (tratamiento especial en tokenizacion).
    dense_features : list[str]
        Features densas (binning por cuantiles).
    removed_features : list[str]
        Features eliminadas por ser constantes.
    """
    print("\n--- Preprocesamiento ---")

    # 1. Eliminar features constantes
    kept, removed = remove_constant_features(df, feature_cols)
    print(f"  Features constantes eliminadas: {len(removed)}")
    if removed:
        print(f"    {removed}")

    # 2. Identificar features dispersas vs densas
    sparse_features, dense_features = identify_sparse_features(df, kept)
    print(f"  Features dispersas (>{config.SPARSE_THRESHOLD*100:.0f}% ceros): "
          f"{len(sparse_features)}")
    print(f"  Features densas: {len(dense_features)}")
    print(f"  Features totales retenidas: {len(kept)}")

    return df, kept, sparse_features, dense_features, removed




if __name__ == "__main__":
    from data_loader import load_dataset, split_features_metadata
    df = load_dataset()
    feature_cols, _ = split_features_metadata(df)
    preprocess(df, feature_cols)