"""
Pipeline UEBA para deteccion de accesos no autorizados via redes P2P.

Dataset real: ISCXVPN2016 - Scenario B (Canadian Institute for Cybersecurity)
  https://www.unb.ca/cic/datasets/vpn.html

Escenario: entorno corporativo donde usuarios instalan clientes P2P
(BitTorrent, eMule) o tunelean P2P dentro de VPN para evadir firewalls.

Clases del dataset original mapeadas a etiquetas UEBA:
  label=0 (benigno):   BROWSING, CHAT, STREAMING, MAIL, VOIP, FT
                        VPN-BROWSING, VPN-CHAT, VPN-STREAMING,
                        VPN-MAIL, VPN-VOIP, VPN-FT
  label=1 (P2P ilegal): P2P, VPN-P2P

Reutiliza completamente los modulos existentes de src/:
  src/data_loader.py   - carga y resumen
  src/exploracion.py   - EDA y graficas
  src/preprocessing.py - limpieza de features
  src/tokenizer.py     - discretizacion adaptativa + vocabulario BoW
  src/bow_builder.py   - representaciones BoW, TF-IDF, nivel usuario
  src/models.py        - clasificadores supervisados + deteccion de anomalias
  src/evaluation.py    - metricas, curvas ROC/PR, matrices de confusion

Las figuras y metricas se guardan en results/p2p/ (separado del proyecto original).

Prerrequisito:
  python prepare_p2p_dataset.py

Uso:
  python main_p2p.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Redirigir rutas de salida ANTES de que los modulos de src/ las lean.
# Todos los modulos importan 'from . import config', por lo que modificar
# config aqui afecta a todos ellos sin tocar ningun archivo existente.
# ---------------------------------------------------------------------------
from src import config  # noqa: E402

P2P_RESULTS_DIR = config.PROJECT_ROOT / "results" / "p2p"
config.FIGURES_DIR = P2P_RESULTS_DIR / "figures"
config.METRICS_DIR = P2P_RESULTS_DIR / "metrics"

for _d in [config.FIGURES_DIR, config.METRICS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

P2P_DATASET_PATH = config.PROJECT_ROOT / "P2P_ISCXVPN.parquet"

# ---------------------------------------------------------------------------
# Importaciones del pipeline (usan config ya redirigido)
# ---------------------------------------------------------------------------
from src.data_loader import load_dataset, split_features_metadata
from src.exploracion import run_eda
from src.preprocessing import preprocess
from src.tokenizer import EventTokenizer
from src.bow_builder import build_sample_bow, build_user_bow, apply_tfidf
from src.models import (
    train_evaluate_representation,
    cross_validate_best,
)
from src.evaluation import (
    plot_confusion_matrices, plot_roc_curves, plot_pr_curves,
    plot_metrics_comparison, plot_feature_importance,
    plot_bow_analysis, save_all_metrics,
)
from sklearn.model_selection import GroupShuffleSplit


def dataset_summary_p2p(df):
    """Resumen especifico del dataset ISCXVPN2016 P2P."""
    feat_cols, _ = split_features_metadata(df)
    vc = df[config.LABEL_COL].value_counts().sort_index()

    print("=" * 70)
    print("DATASET: ISCXVPN2016 - Accesos no autorizados via P2P")
    print("Fuente: Canadian Institute for Cybersecurity - Scenario B")
    print("=" * 70)
    print(f"Muestras totales:       {len(df):,}")
    print(f"Features numericas:     {len(feat_cols)}")
    print(f"Usuarios unicos:        {df[config.USER_COL].nunique()}")
    print(f"Rango temporal:         {df[config.TIMESTAMP_COL].min()} - "
          f"{df[config.TIMESTAMP_COL].max()}")
    print()
    print("Distribucion de clases:")
    for lbl, count in vc.items():
        pct = 100 * count / len(df)
        tag = "Benigno (corp.)" if lbl == 0 else "P2P no autorizado"
        print(f"  {tag} (label={lbl}): {count:>7,} ({pct:.2f}%)")
    ratio = vc[0] / vc[1]
    print(f"\nRatio de desbalance:    {ratio:.1f}:1")
    print()
    print("Usuarios por clase:")
    user_lbl = df.groupby(config.USER_COL)[config.LABEL_COL].max()
    print(f"  Benignos : {(user_lbl==0).sum()} usuarios")
    print(f"  P2P      : {(user_lbl==1).sum()} usuarios")

    if "traffic_class" in df.columns:
        print()
        print("Distribucion de trafico original (top 6):")
        tc_vc = df["traffic_class"].value_counts().head(6)
        for tc, cnt in tc_vc.items():
            pct = 100 * cnt / len(df)
            print(f"  {tc:<15}: {cnt:>6,} ({pct:.1f}%)")
    print("=" * 70)


def main():
    t0 = time.time()
    print("=" * 70)
    print("PROYECTO: Extraccion de caracteristicas para UEBA")
    print("         usando frequency counting (bag-of-words)")
    print("Dataset:  ISCXVPN2016 - Scenario B")
    print("Escenario: Accesos no autorizados via P2P (BitTorrent / VPN-P2P)")
    print("Fuente:   Canadian Institute for Cybersecurity")
    print("Unidad:   Entidad (workstation)")
    print("=" * 70)

    if not P2P_DATASET_PATH.exists():
        print(f"\n[ERROR] No se encontro el dataset en: {P2P_DATASET_PATH}")
        print("  Ejecuta primero: python prepare_p2p_dataset.py")
        sys.exit(1)

    # ==================================================================
    # PASO 1: Carga de datos
    # ==================================================================
    print("\n[1/9] Carga del dataset")
    df = load_dataset(path=P2P_DATASET_PATH)
    dataset_summary_p2p(df)
    # Eliminar columnas de metadato extra del adaptador (no son features numericas)
    df = df.drop(columns=[c for c in ["traffic_class", "time_window"] if c in df.columns])
    feature_cols, meta_cols = split_features_metadata(df)
    print(f"  Features numericas cargadas: {len(feature_cols)}")
    dataset_summary_p2p(df)

    # ==================================================================
    # PASO 2: EDA
    # ==================================================================
    print("\n[2/9] Analisis exploratorio")
    run_eda(df, feature_cols)

    # ==================================================================
    # PASO 3: Preprocesamiento
    # ==================================================================
    print("\n[3/9] Preprocesamiento")
    df, clean_features, sparse_features, dense_features, removed = preprocess(
        df, feature_cols
    )

    # ==================================================================
    # PASO 4: Tokenizacion
    # ==================================================================
    print("\n[4/9] Tokenizacion")
    print(f"  Features densas:    {len(dense_features)}")
    print(f"  Features dispersas: {len(sparse_features)}")

    # Ajustar tokenizador SOLO sobre ventanas benignas
    benign_df = df[df[config.LABEL_COL] == 0]
    tokenizer = EventTokenizer(num_bins=config.NUM_BINS)
    tokenizer.fit(benign_df, sparse_features, dense_features)

    info = tokenizer.get_vocabulary_info()
    print(f"  Vocabulario: {info['vocab_size']} tokens")
    print(f"  Ejemplos: {tokenizer.vocabulary_[:3]}")

    # ==================================================================
    # PASO 5: Representaciones
    # ==================================================================
    print("\n[5/9] Construccion de representaciones")

    bow_raw, labels, user_ids = build_sample_bow(df, tokenizer)

    print("\n  Aplicando TF-IDF sobre BoW...")
    bow_tfidf = apply_tfidf(bow_raw)
    print(f"  Matriz TF-IDF: {bow_tfidf.shape}")

    X_raw = df[clean_features].values.astype(np.float32)
    print(f"\n  Features originales (baseline): {X_raw.shape}")

    user_bow, user_labels, user_list = build_user_bow(df, tokenizer)

    # ==================================================================
    # PASO 6: Division train/test
    # ==================================================================
    print("\n[6/9] Division train/test")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(bow_raw, labels, groups=user_ids))

    y_train = labels[train_idx]
    y_test = labels[test_idx]
    u_train = user_ids[train_idx]
    u_test = user_ids[test_idx]

    print(f"  Train: {len(train_idx):,} muestras, {len(set(u_train))} usuarios "
          f"({(y_train==1).sum():,} positivos)")
    print(f"  Test:  {len(test_idx):,} muestras, {len(set(u_test))} usuarios "
          f"({(y_test==1).sum():,} positivos)")
    assert not (set(u_train) & set(u_test)), "Filtracion de usuarios detectada"
    print("  Sin filtracion de usuarios entre train/test")

    splits = {
        "BoW (frecuencias)": (bow_raw[train_idx], bow_raw[test_idx]),
        "BoW + TF-IDF":      (bow_tfidf[train_idx], bow_tfidf[test_idx]),
        "Features originales": (X_raw[train_idx], X_raw[test_idx]),
    }

    # ==================================================================
    # PASO 7: Clasificacion supervisada
    # ==================================================================
    print("\n[7/8] Clasificacion supervisada (3 representaciones)")
    all_results = {}

    for rep_name, (X_tr, X_te) in splits.items():
        use_sc = (rep_name == "Features originales")
        results, best, preds = train_evaluate_representation(
            X_tr, X_te, y_train, y_test,
            rep_name=rep_name,
            use_scaling=use_sc,
            downsample_ratio=1.0,
        )
        all_results[rep_name] = (results, best, preds)

    # ==================================================================
    # PASO 8: Evaluacion y visualizacion
    # ==================================================================
    print("\n[8/8] Evaluacion e interpretabilidad")

    n_pos_test = int((y_test == 1).sum())
    y_test_bal = np.concatenate([
        np.zeros(n_pos_test, dtype=int), np.ones(n_pos_test, dtype=int)
    ])

    bow_results, bow_best, bow_preds = all_results["BoW + TF-IDF"]
    plot_confusion_matrices(bow_results, prefix="06",
                            title_extra=" - BoW+TF-IDF (P2P ISCXVPN)")
    plot_roc_curves(bow_results, bow_preds, y_test_bal, prefix="07",
                    title_extra=" - BoW+TF-IDF (P2P ISCXVPN)")
    plot_pr_curves(bow_results, bow_preds, y_test_bal, prefix="08",
                   title_extra=" - BoW+TF-IDF (P2P ISCXVPN)")
    print("  [OK] Graficas de clasificacion BoW + TF-IDF")

    raw_results, raw_best, raw_preds = all_results["Features originales"]
    plot_confusion_matrices(raw_results, prefix="06b",
                            title_extra=" - Features originales (P2P ISCXVPN)")
    plot_roc_curves(raw_results, raw_preds, y_test_bal, prefix="07b",
                    title_extra=" - Features originales (P2P ISCXVPN)")
    print("  [OK] Graficas de clasificacion Features originales")

    plot_metrics_comparison(all_results)
    print("  [OK] Comparacion de representaciones")

    top_tokens = plot_feature_importance(
        bow_best[1], bow_best[0], tokenizer.vocabulary_
    )
    print("  [OK] Importancia de tokens")

    plot_bow_analysis(bow_raw, labels, tokenizer.vocabulary_)
    print("  [OK] Analisis BoW")

    df_metrics = save_all_metrics(all_results)

    # ==================================================================
    # Validacion cruzada
    # ==================================================================
    print("\n--- Validacion cruzada del mejor modelo ---")
    cv_metrics = cross_validate_best(
        bow_tfidf, labels, user_ids, bow_best[0], n_splits=5
    )

    # ==================================================================
    # RESUMEN FINAL
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("PIPELINE P2P (ISCXVPN2016) COMPLETADO")
    print(f"{'='*70}")
    print(f"  Dataset:    ISCXVPN2016 Scenario B ({len(df):,} muestras)")
    print(f"  Tiempo:     {elapsed:.0f}s")
    print(f"  Vocabulario BoW: {info['vocab_size']} tokens")
    print(f"  Documentos: {bow_raw.shape[0]:,} muestras / {user_bow.shape[0]} usuarios")
    print()
    print("  Mejor modelo por representacion:")
    for rep_name, (results, best, _) in all_results.items():
        f1 = results[best[0]]["F1"]
        roc = results[best[0]]["ROC-AUC"]
        print(f"    {rep_name}: {best[0]} (F1={f1:.4f}, ROC-AUC={roc:.4f})")

    if top_tokens:
        print(f"\n  Top 5 tokens P2P mas discriminativos ({bow_best[0]}):")
        for token, imp in top_tokens[:5]:
            print(f"    {token}: {imp:.4f}")

    print(f"\n  Figuras:  {config.FIGURES_DIR}")
    print(f"  Metricas: {config.METRICS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
