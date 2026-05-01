"""
Evaluacion, visualizacion de resultados e interpretabilidad.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay,
)
from . import config


def plot_confusion_matrices(results, prefix="06", title_extra="", save=True):
    """Matrices de confusion."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, results.items()):
        ConfusionMatrixDisplay(m["Confusion Matrix"],
                               display_labels=["Benigno", "Riesgo"]).plot(
            ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{name}\nF1={m['F1']:.3f}", fontsize=9)
    plt.suptitle(f"Matrices de Confusion{title_extra}", fontsize=11, y=1.05)
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / f"{prefix}_matrices_confusion.png")
    plt.close(fig)


def plot_roc_curves(results, predictions, y_test, prefix="07",
                    title_extra="", save=True):
    """Curvas ROC comparativas."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    for (name, m), color in zip(results.items(), colors):
        y_p = predictions[name]["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={m['ROC-AUC']:.3f})",
                color=color, linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"Curvas ROC{title_extra}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / f"{prefix}_curvas_roc.png")
    plt.close(fig)


def plot_pr_curves(results, predictions, y_test, prefix="08",
                   title_extra="", save=True):
    """Curvas Precision-Recall."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    baseline = (y_test == 1).sum() / len(y_test)
    for (name, m), color in zip(results.items(), colors):
        y_p = predictions[name]["y_proba"]
        prec, rec, _ = precision_recall_curve(y_test, y_p)
        ax.plot(rec, prec, label=f"{name} (AP={m['PR-AUC']:.3f})",
                color=color, linewidth=1.5)
    ax.axhline(y=baseline, color="gray", ls="--", alpha=0.5,
               label=f"Baseline ({baseline:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Curvas Precision-Recall{title_extra}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / f"{prefix}_curvas_pr.png")
    plt.close(fig)


def plot_metrics_comparison(all_rep_results, save=True):
    """Comparacion de metricas entre representaciones y clasificadores."""
    metric_keys = ["F1", "F1_macro", "Precision", "Recall (TPR)",
                   "ROC-AUC", "PR-AUC"]

    rows = []
    for rep_name, (results, _, _) in all_rep_results.items():
        for clf_name, m in results.items():
            for mk in metric_keys:
                rows.append({
                    "Representacion": rep_name,
                    "Clasificador": clf_name,
                    "Metrica": mk,
                    "Valor": m.get(mk, 0),
                })
    df = pd.DataFrame(rows)

    # Encontrar mejor clasificador por F1 para cada representacion
    best_per_rep = {}
    for rep_name, (results, best_info, _) in all_rep_results.items():
        best_per_rep[rep_name] = best_info[0]

    # Grafica: mejor modelo de cada representacion vs metricas
    fig, ax = plt.subplots(figsize=(12, 5))
    reps = list(all_rep_results.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / len(reps)
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(reps)))

    for i, rep in enumerate(reps):
        best_clf = best_per_rep[rep]
        results = all_rep_results[rep][0]
        vals = [results[best_clf].get(mk, 0) for mk in metric_keys]
        offset = (i - len(reps) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=f"{rep} ({best_clf})", color=colors[i],
                      edgecolor="black", linewidth=0.3)
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6,
                        rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=9)
    ax.set_ylabel("Valor")
    ax.set_title("Comparacion: Mejor modelo por representacion")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "09_comparacion_representaciones.png")
    plt.close(fig)


def plot_feature_importance(model, model_name, vocabulary, top_n=20, save=True):
    """Importancia de features (tokens) para modelos de arbol."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Para Logistic Regression: usar valor absoluto de coeficientes
        imp = np.abs(model.coef_[0])
    else:
        print(f"  {model_name} no soporta importancia de features")
        return None

    n_feats = len(imp)
    top_idx = np.argsort(imp)[-top_n:][::-1]
    top_tokens = [vocabulary[i] if i < len(vocabulary) else f"feat_{i}"
                  for i in top_idx]
    top_vals = imp[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(top_tokens))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top_tokens)))
    ax.barh(y_pos, top_vals, color=colors[::-1], edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_tokens, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia")
    ax.set_title(f"Top {top_n} Tokens - {model_name}")
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "10_importancia_tokens.png")
    plt.close(fig)

    return list(zip(top_tokens, top_vals))


def plot_bow_analysis(bow_matrix, labels, vocabulary, save=True):
    """Analisis de la representacion BoW."""
    b_mask = labels == 0
    r_mask = labels == 1

    freq_b = bow_matrix[b_mask].mean(axis=0)
    freq_r = bow_matrix[r_mask].mean(axis=0)
    diff = np.abs(freq_r - freq_b)
    top_idx = np.argsort(diff)[-15:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top_tokens = [vocabulary[i] for i in top_idx]
    y_pos = np.arange(len(top_tokens))
    w = 0.35
    axes[0].barh(y_pos - w/2, freq_b[top_idx], w,
                 label="Benigno", color="#2ecc71", alpha=0.8)
    axes[0].barh(y_pos + w/2, freq_r[top_idx], w,
                 label="Riesgo", color="#e74c3c", alpha=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_tokens, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Frecuencia media")
    axes[0].set_title("Tokens con mayor diferencia entre clases")
    axes[0].legend(fontsize=8)

    tps_b = bow_matrix[b_mask].sum(axis=1)
    tps_r = bow_matrix[r_mask].sum(axis=1)
    axes[1].hist(tps_b, bins=30, alpha=0.6, label="Benigno", color="#2ecc71",
                 density=True)
    axes[1].hist(tps_r, bins=30, alpha=0.6, label="Riesgo", color="#e74c3c",
                 density=True)
    axes[1].set_xlabel("Tokens activos por muestra")
    axes[1].set_ylabel("Densidad")
    axes[1].set_title("Distribucion de tokens activos")
    axes[1].legend(fontsize=8)

    plt.suptitle("Analisis de la Representacion Bag-of-Words", fontsize=12, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "13_analisis_bow.png")
    plt.close(fig)


def save_all_metrics(all_rep_results, save=True):
    """Guarda tabla consolidada de todas las metricas."""
    rows = []
    for rep_name, (results, _, _) in all_rep_results.items():
        for clf_name, m in results.items():
            row = {"Representacion": rep_name, "Modelo": clf_name,
                   "Tipo": "Supervisado"}
            for key in ["F1", "F1_macro", "Precision", "Recall (TPR)",
                         "Accuracy", "ROC-AUC", "PR-AUC", "TPR", "FPR"]:
                row[key] = m.get(key, None)
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print("TABLA CONSOLIDADA DE RESULTADOS")
    print(f"{'='*70}")
    # Mostrar solo las columnas mas relevantes
    show_cols = ["Representacion", "Modelo", "Tipo", "F1", "F1_macro",
                 "Precision", "Recall (TPR)", "ROC-AUC", "PR-AUC"]
    existing = [c for c in show_cols if c in df.columns]
    print(df[existing].to_string(index=False, float_format="%.4f"))

    if save:
        path = config.METRICS_DIR / "resultados_consolidados.csv"
        df.to_csv(path, index=False, float_format="%.4f")
        print(f"\n  Guardado en: {path}")
    return df
