"""
Pipeline principal: Extraccion de caracteristicas para UEBA usando bag-of-words.
Dataset: RBD24 - Crypto Desktop (deteccion de actividad de criptomineria).

Unidad de analisis: entidad (workstation de escritorio).
Ventana temporal: 1 hora con desplazamiento de 10 minutos (predefinido en RBD24).

El pipeline evalua tres representaciones:
  1. BoW cruda (conteo de frecuencias de tokens discretizados).
  2. BoW + TF-IDF (ponderacion por relevancia).
  3. Features originales (baseline de comparacion).

Estrategia de balanceo: downsampling de clase mayoritaria (como el paper RBD24).
Division: train/test a nivel de usuario (sin filtracion de datos).
"""
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src import config
from src.data_loader import load_dataset, split_features_metadata, dataset_summary
from src.exploracion import run_eda
from src.preprocessing import preprocess
from src.tokenizer import EventTokenizer
from src.bow_builder import build_sample_bow, build_user_bow, apply_tfidf
from src.models import (
    user_level_split, train_evaluate_representation,
    cross_validate_best,
)
from src.evaluation import (
    plot_confusion_matrices, plot_roc_curves, plot_pr_curves,
    plot_metrics_comparison, plot_feature_importance,
    plot_bow_analysis, save_all_metrics,
)


def main():
    t0 = time.time()
    print("=" * 70)
    print("PROYECTO: Extraccion de caracteristicas para UEBA")
    print("         usando frequency counting (bag-of-words)")
    print("Dataset:  RBD24 - Crypto Desktop")
    print("Unidad:   Entidad (workstation)")
    print("Ventana:  1 hora, stride 10 min")
    print("=" * 70)

    # ==================================================================
    # PASO 1: Carga de datos
    # ==================================================================
    print("\n[1/9] Carga del dataset")
    df = load_dataset()
    feature_cols, meta_cols = split_features_metadata(df)
    dataset_summary(df, feature_cols)

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
    print(f"  Features densas:   {len(dense_features)}")
    print(f"  Features dispersas: {len(sparse_features)}")

    # Ajustar tokenizador SOLO con datos benignos
    # (captura perfil de comportamiento normal, recomendacion del asesor)
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

    # 5a. BoW (nivel muestra)
    bow_raw, labels, user_ids = build_sample_bow(df, tokenizer)

    # 5b. BoW + TF-IDF
    print("\n  Aplicando TF-IDF sobre BoW...")
    bow_tfidf = apply_tfidf(bow_raw)
    print(f"  Matriz TF-IDF: {bow_tfidf.shape}")

    # 5c. Features originales (baseline)
    X_raw = df[clean_features].values.astype(np.float32)
    print(f"\n  Features originales (baseline): {X_raw.shape}")

    # 5d. BoW a nivel usuario (analisis adicional)
    user_bow, user_labels, user_list = build_user_bow(df, tokenizer)

    # ==================================================================
    # PASO 6: Division train/test
    # ==================================================================
    print("\n[6/9] Division train/test")
    # Usar la misma division para todas las representaciones
    from sklearn.model_selection import GroupShuffleSplit
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
    assert not (set(u_train) & set(u_test))
    print("  Sin filtracion de usuarios")

    # Preparar splits para cada representacion
    splits = {
        "BoW (frecuencias)": (bow_raw[train_idx], bow_raw[test_idx]),
        "BoW + TF-IDF": (bow_tfidf[train_idx], bow_tfidf[test_idx]),
        "Features originales": (X_raw[train_idx], X_raw[test_idx]),
    }

    # ==================================================================
    # PASO 7: Clasificacion supervisada (3 representaciones)
    # ==================================================================
    print("\n[7/8] Clasificacion supervisada")
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
    # PASO 8: Evaluacion, visualizacion e interpretabilidad
    # ==================================================================
    print("\n[8/8] Evaluacion e interpretabilidad")

    # Crear y_test balanceado para las graficas
    from imblearn.under_sampling import RandomUnderSampler
    n_pos_test = int((y_test == 1).sum())
    test_sampler = RandomUnderSampler(
        sampling_strategy={0: n_pos_test, 1: n_pos_test},
        random_state=config.RANDOM_STATE,
    )
    # Solo necesitamos los labels balanceados para las graficas
    # Las predicciones ya estan hechas sobre el test balanceado
    y_test_bal = np.concatenate([
        np.zeros(n_pos_test, dtype=int), np.ones(n_pos_test, dtype=int)
    ])

    # Generar graficas para la representacion BoW + TF-IDF (principal)
    bow_results, bow_best, bow_preds = all_results["BoW + TF-IDF"]
    plot_confusion_matrices(bow_results, prefix="06",
                            title_extra=" - BoW + TF-IDF")
    plot_roc_curves(bow_results, bow_preds, y_test_bal, prefix="07",
                    title_extra=" - BoW + TF-IDF")
    plot_pr_curves(bow_results, bow_preds, y_test_bal, prefix="08",
                   title_extra=" - BoW + TF-IDF")
    print("  [OK] Graficas de clasificacion BoW + TF-IDF")

    # Graficas para features originales
    raw_results, raw_best, raw_preds = all_results["Features originales"]
    plot_confusion_matrices(raw_results, prefix="06b",
                            title_extra=" - Features originales")
    plot_roc_curves(raw_results, raw_preds, y_test_bal, prefix="07b",
                    title_extra=" - Features originales")
    print("  [OK] Graficas de clasificacion Features originales")

    # Comparacion entre representaciones
    plot_metrics_comparison(all_results)
    print("  [OK] Comparacion de representaciones")

    # Importancia de tokens (mejor modelo BoW)
    top_tokens = plot_feature_importance(
        bow_best[1], bow_best[0], tokenizer.vocabulary_
    )
    print("  [OK] Importancia de tokens")

    # Analisis BoW
    plot_bow_analysis(bow_raw, labels, tokenizer.vocabulary_)
    print("  [OK] Analisis BoW")

    # Tabla consolidada
    df_metrics = save_all_metrics(all_results)

    # ==================================================================
    # Validacion cruzada del mejor modelo BoW
    # ==================================================================
    print("\n--- Validacion cruzada del mejor modelo ---")
    best_bow_name = bow_best[0]
    cv_metrics = cross_validate_best(
        bow_tfidf, labels, user_ids, best_bow_name, n_splits=5
    )

    # ==================================================================
    # RESUMEN FINAL
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETADO")
    print(f"{'='*70}")
    print(f"  Tiempo: {elapsed:.0f}s")
    print(f"  Vocabulario BoW: {info['vocab_size']} tokens")
    print(f"  Documentos: {bow_raw.shape[0]:,} muestras / {user_bow.shape[0]} usuarios")
    print()

    print("  Mejor modelo por representacion:")
    for rep_name, (results, best, _) in all_results.items():
        f1 = results[best[0]]["F1"]
        roc = results[best[0]]["ROC-AUC"]
        print(f"    {rep_name}: {best[0]} (F1={f1:.4f}, ROC-AUC={roc:.4f})")

    if top_tokens:
        print(f"\n  Top 5 tokens mas discriminativos ({bow_best[0]}):")
        for token, imp in top_tokens[:5]:
            print(f"    {token}: {imp:.4f}")

    print(f"\n  Figuras: {config.FIGURES_DIR}")
    print(f"  Metricas: {config.METRICS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
