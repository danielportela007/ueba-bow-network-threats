
"""
Construccion de la representacion bag-of-words con multiples variantes:
- BoW con conteo de frecuencias (nivel muestra).
- BoW con conteo de frecuencias (nivel usuario - agregacion temporal).
- Normalizacion TF-IDF.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as sk_normalize
from . import config


def build_sample_bow(df, tokenizer):
    """Genera la representacion BoW a nivel de muestra (cada ventana = 1 doc)."""
    print("\n--- BoW nivel muestra ---")
    bow = tokenizer.transform(df)
    labels = df[config.LABEL_COL].values
    user_ids = df[config.USER_COL].values

    info = tokenizer.get_vocabulary_info()
    print(f"  Vocabulario: {info['vocab_size']} tokens")
    print(f"  Matriz BoW:  {bow.shape}")
    print(f"  Tokens activos/doc (media): {bow.sum(axis=1).mean():.1f}")
    return bow, labels, user_ids


def build_user_bow(df, tokenizer):
    """Genera BoW a nivel de usuario (conteo de frecuencias genuino)."""
    print("\n--- BoW nivel usuario (agregacion temporal) ---")
    sample_bow = tokenizer.transform(df)
    user_ids = df[config.USER_COL].values
    labels = df[config.LABEL_COL].values

    unique_users = sorted(set(user_ids))
    V = sample_bow.shape[1]
    user_bow = np.zeros((len(unique_users), V), dtype=np.float32)
    user_labels = np.zeros(len(unique_users), dtype=np.int32)

    uid_map = {u: i for i, u in enumerate(unique_users)}
    for i in range(len(df)):
        idx = uid_map[user_ids[i]]
        user_bow[idx] += sample_bow[i]
        user_labels[idx] = max(user_labels[idx], labels[i])

    print(f"  Usuarios: {len(unique_users)} ({(user_labels==1).sum()} riesgo)")
    print(f"  Frecuencia media de tokens/usuario: {user_bow.sum(axis=1).mean():.0f}")
    return user_bow, user_labels, unique_users


def apply_tfidf(bow_matrix):
    """Aplica ponderacion TF-IDF a la matriz BoW.

    TF = freq del token / total de tokens en el documento
    IDF = log((N + 1) / (df + 1)) + 1  (smooth IDF)

    Esto reduce el peso de tokens muy comunes (presentes en casi todos
    los documentos) y realza tokens raros/discriminativos.
    """
    N = bow_matrix.shape[0]

    # TF: normalizacion por documento
    row_sums = bow_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    tf = bow_matrix / row_sums

    # IDF: frecuencia inversa de documentos
    df_counts = (bow_matrix > 0).sum(axis=0).astype(np.float32)
    idf = np.log((N + 1.0) / (df_counts + 1.0)) + 1.0

    tfidf = tf * idf

    # Normalizacion L2 por fila
    tfidf = sk_normalize(tfidf, norm="l2")
    return tfidf



if __name__ == "__main__":
    from data_loader import load_dataset, split_features_metadata
    from preprocessing import preprocess
    from tokenizer import EventTokenizer

    df = load_dataset()
    feature_cols, _ = split_features_metadata(df)
    df, kept, sparse_features, dense_features, removed = preprocess(df, feature_cols)

    tokenizer = EventTokenizer()
    tokenizer.fit(df, sparse_features, dense_features)

    bow, labels, user_ids = build_sample_bow(df, tokenizer)
    user_bow, user_labels, unique_users = build_user_bow(df, tokenizer)
    tfidf = apply_tfidf(bow)
    print(f"Matriz TF-IDF: {tfidf.shape}")