
"""
Tokenizacion de features numericas para representacion bag-of-words.

Estrategia de discretizacion adaptativa:
- Features densas: binning por cuantiles.
- Features dispersas (>90% ceros): categoria 'cero' + bins de no-cero.

Cada feature se convierte en un token "{feature_name}={bin_label}".
El tokenizador se ajusta sobre datos de entrenamiento (idealmente solo benignos)
para capturar el perfil de comportamiento normal.
"""
import numpy as np
import pandas as pd
from . import config


class EventTokenizer:
    """Tokenizador adaptativo de features numericas."""

    def __init__(self, num_bins=None, sparse_threshold=None):
        self.num_bins = num_bins or config.NUM_BINS
        self.sparse_threshold = sparse_threshold or config.SPARSE_THRESHOLD
        self.bin_edges_ = {}
        self.sparse_cols_ = []
        self.dense_cols_ = []
        self.vocabulary_ = []
        self.token_to_idx_ = {}
        self.is_fitted_ = False

    def fit(self, df, sparse_features, dense_features):
        """Aprende limites de binning a partir de datos de entrenamiento."""
        self.sparse_cols_ = list(sparse_features)
        self.dense_cols_ = list(dense_features)
        self.bin_edges_ = {}

        for col in self.dense_cols_:
            values = df[col].dropna().values
            quantiles = np.linspace(0, 1, self.num_bins + 1)
            edges = np.unique(np.quantile(values, quantiles))
            if len(edges) < 3:
                edges = np.unique([values.min(), np.median(values), values.max()])
            self.bin_edges_[col] = edges

        for col in self.sparse_cols_:
            values = df[col].dropna().values
            nonzero = values[values != 0]
            if len(nonzero) > 10:
                n_nz = max(2, self.num_bins - 1)
                edges = np.unique(np.quantile(nonzero, np.linspace(0, 1, n_nz + 1)))
                if len(edges) < 2:
                    edges = np.unique([nonzero.min(), nonzero.max()])
            elif len(nonzero) > 0:
                edges = np.unique([nonzero.min(), nonzero.max()])
            else:
                edges = np.array([0.0])
            self.bin_edges_[col] = edges

        self._build_vocabulary()
        self.is_fitted_ = True
        return self

    @staticmethod
    def _make_labels(n):
        maps = {
            1: ["unico"], 2: ["bajo", "alto"], 3: ["bajo", "medio", "alto"],
            4: ["bajo", "medio_bajo", "medio_alto", "alto"],
            5: ["muy_bajo", "bajo", "medio", "alto", "muy_alto"],
        }
        return maps.get(n, [f"bin_{i}" for i in range(n)])

    def _build_vocabulary(self):
        vocab = set()
        for col in self.dense_cols_:
            n = max(1, len(self.bin_edges_[col]) - 1)
            for lbl in self._make_labels(n):
                vocab.add(f"{col}={lbl}")
        for col in self.sparse_cols_:
            vocab.add(f"{col}=cero")
            edges = self.bin_edges_[col]
            if len(edges) <= 1:
                vocab.add(f"{col}=presente")
            else:
                n = max(1, len(edges) - 1)
                for lbl in self._make_labels(n):
                    vocab.add(f"{col}={lbl}")
        self.vocabulary_ = sorted(vocab)
        self.token_to_idx_ = {t: i for i, t in enumerate(self.vocabulary_)}

    def transform(self, df):
        """Genera la matriz documento-termino (BoW) a partir del DataFrame."""
        if not self.is_fitted_:
            raise RuntimeError("Tokenizador no ajustado. Llamar fit() primero.")

        n = len(df)
        V = len(self.vocabulary_)
        bow = np.zeros((n, V), dtype=np.float32)

        for col in self.dense_cols_ + self.sparse_cols_:
            vals = df[col].fillna(0).values
            edges = self.bin_edges_[col]

            if col in self.sparse_cols_:
                is_zero = vals == 0
                tz = f"{col}=cero"
                if tz in self.token_to_idx_:
                    bow[is_zero, self.token_to_idx_[tz]] = 1
                nz_mask = ~is_zero
                if nz_mask.any():
                    if len(edges) <= 1:
                        tp = f"{col}=presente"
                        if tp in self.token_to_idx_:
                            bow[nz_mask, self.token_to_idx_[tp]] = 1
                    else:
                        nz_vals = vals[nz_mask]
                        bi = np.clip(np.searchsorted(edges, nz_vals, side="right") - 1,
                                     0, len(edges) - 2)
                        labels = self._make_labels(max(1, len(edges) - 1))
                        nz_idx = np.where(nz_mask)[0]
                        for k in range(len(labels)):
                            t = f"{col}={labels[k]}"
                            if t in self.token_to_idx_:
                                bow[nz_idx[bi == k], self.token_to_idx_[t]] = 1
            else:
                if len(edges) <= 1:
                    t = f"{col}=unico"
                    if t in self.token_to_idx_:
                        bow[:, self.token_to_idx_[t]] = 1
                else:
                    bi = np.clip(np.searchsorted(edges, vals, side="right") - 1,
                                 0, len(edges) - 2)
                    labels = self._make_labels(max(1, len(edges) - 1))
                    for k in range(len(labels)):
                        t = f"{col}={labels[k]}"
                        if t in self.token_to_idx_:
                            bow[bi == k, self.token_to_idx_[t]] = 1
        return bow

    def get_vocabulary_info(self):
        return {
            "vocab_size": len(self.vocabulary_),
            "n_dense_features": len(self.dense_cols_),
            "n_sparse_features": len(self.sparse_cols_),
            "n_total_features": len(self.dense_cols_) + len(self.sparse_cols_),
        }



if __name__ == "__main__":
    from data_loader import load_dataset, split_features_metadata
    from preprocessing import preprocess

    df = load_dataset()
    feature_cols, _ = split_features_metadata(df)
    df, kept, sparse_features, dense_features, removed = preprocess(df, feature_cols)

    tokenizer = EventTokenizer()
    tokenizer.fit(df, sparse_features, dense_features)
    print("Vocabulario generado:", tokenizer.get_vocabulary_info())