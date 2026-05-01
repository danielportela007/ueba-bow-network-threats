"""
Analisis exploratorio de datos (EDA) y visualizaciones.
Genera graficas del comportamiento del dataset RBD24 Crypto Desktop.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from . import config


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
})


def plot_class_distribution(df, save=True):
    """Grafica de barras de la distribucion de clases."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Nivel muestra
    vc = df[config.LABEL_COL].value_counts().sort_index()
    labels_map = {0: "Benigno", 1: "Riesgo\n(Crypto)"}
    names = [labels_map[i] for i in vc.index]
    colors = ["#2ecc71", "#e74c3c"]
    bars = axes[0].bar(names, vc.values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vc.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,}", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Distribucion de clases (nivel muestra)")
    axes[0].set_ylabel("Numero de muestras")
    axes[0].set_yscale("log")

    # Nivel usuario
    user_labels = df.groupby(config.USER_COL)[config.LABEL_COL].max()
    vc_u = user_labels.value_counts().sort_index()
    names_u = [labels_map[i] for i in vc_u.index]
    bars = axes[1].bar(names_u, vc_u.values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vc_u.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Distribucion de clases (nivel usuario)")
    axes[1].set_ylabel("Numero de usuarios")

    plt.suptitle("Desbalance de clases - Dataset Crypto Desktop", fontsize=12, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "01_distribucion_clases.png")
    plt.close(fig)


def plot_feature_distributions(df, feature_cols, top_n=12, save=True):
    """Histogramas comparativos de las features mas discriminativas."""
    benign = df[df[config.LABEL_COL] == 0]
    risk = df[df[config.LABEL_COL] == 1]

    # Seleccionar features con mayor diferencia en medias normalizadas
    diffs = []
    for col in feature_cols:
        std_total = df[col].std()
        if std_total > 0:
            diff = abs(benign[col].mean() - risk[col].mean()) / std_total
            diffs.append((col, diff))
    diffs.sort(key=lambda x: x[1], reverse=True)
    top_features = [d[0] for d in diffs[:top_n]]

    ncols = 3
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(top_features):
        ax = axes[i]
        ax.hist(benign[col].clip(lower=benign[col].quantile(0.01),
                                  upper=benign[col].quantile(0.99)),
                bins=50, alpha=0.6, label="Benigno", color="#2ecc71", density=True)
        ax.hist(risk[col].clip(lower=risk[col].quantile(0.01),
                                upper=risk[col].quantile(0.99)),
                bins=50, alpha=0.6, label="Riesgo", color="#e74c3c", density=True)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Features con mayor separacion entre clases", fontsize=12, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "02_top_features_distribucion.png")
    plt.close(fig)
    return top_features


def plot_temporal_patterns(df, save=True):
    """Patron temporal de actividad para muestras benignas vs riesgo."""
    df_temp = df.copy()
    df_temp["hour"] = df_temp[config.TIMESTAMP_COL].dt.hour
    df_temp["day_of_week"] = df_temp[config.TIMESTAMP_COL].dt.dayofweek

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Actividad por hora
    for label, color, name in [(0, "#2ecc71", "Benigno"), (1, "#e74c3c", "Riesgo")]:
        subset = df_temp[df_temp[config.LABEL_COL] == label]
        hourly = subset.groupby("hour").size()
        hourly = hourly / hourly.sum()
        axes[0].plot(hourly.index, hourly.values, marker="o", markersize=3,
                     label=name, color=color, linewidth=1.5)
    axes[0].set_xlabel("Hora del dia")
    axes[0].set_ylabel("Proporcion de muestras")
    axes[0].set_title("Patron horario de actividad")
    axes[0].legend()
    axes[0].set_xticks(range(0, 24))

    # Actividad por dia de la semana
    day_names = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
    for label, color, name in [(0, "#2ecc71", "Benigno"), (1, "#e74c3c", "Riesgo")]:
        subset = df_temp[df_temp[config.LABEL_COL] == label]
        daily = subset.groupby("day_of_week").size()
        daily = daily / daily.sum()
        axes[1].plot(daily.index, daily.values, marker="s", markersize=4,
                     label=name, color=color, linewidth=1.5)
    axes[1].set_xlabel("Dia de la semana")
    axes[1].set_ylabel("Proporcion de muestras")
    axes[1].set_title("Patron semanal de actividad")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names)
    axes[1].legend()

    plt.suptitle("Patrones temporales - Benigno vs Riesgo", fontsize=12, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "03_patrones_temporales.png")
    plt.close(fig)


def plot_correlation_heatmap(df, feature_cols, save=True):
    """Heatmap de correlacion entre grupos de features."""
    # Seleccionar un subconjunto representativo para legibilidad
    groups = {
        "dns": [c for c in feature_cols if c.startswith("dns_") and "interlog" not in c],
        "ssl": [c for c in feature_cols if c.startswith("ssl_")],
        "http": [c for c in feature_cols if c.startswith("http_")],
    }
    selected = []
    for g_cols in groups.values():
        selected.extend(g_cols[:8])  # max 8 por grupo

    if len(selected) < 5:
        return

    corr = df[selected].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax,
                xticklabels=True, yticklabels=True,
                linewidths=0.3, fmt=".1f", annot=False)
    ax.tick_params(labelsize=6)
    ax.set_title("Correlacion entre features seleccionadas", fontsize=12)
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "04_correlacion_features.png")
    plt.close(fig)


def plot_normal_behavior_profile(df, feature_cols, save=True):
    """Perfil de comportamiento normal: distribucion de features en clase benigna.

    Analiza la distribucion de las features mas informativas en la clase benigna
    para entender el comportamiento tipico (recomendacion del asesor).
    """
    benign = df[df[config.LABEL_COL] == 0]
    risk = df[df[config.LABEL_COL] == 1]

    # Features con mayor varianza en clase benigna
    variances = benign[feature_cols].var().sort_values(ascending=False)
    non_zero_var = variances[variances > 0]
    top_var_features = non_zero_var.head(9).index.tolist()

    if not top_var_features:
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(top_var_features):
        ax = axes[i]
        # Boxplot comparativo
        data_benign = benign[col].values
        data_risk = risk[col].values

        # Clip outliers para mejor visualizacion
        q99 = np.percentile(np.concatenate([data_benign, data_risk]), 99)
        data_benign_clip = np.clip(data_benign, 0, q99)
        data_risk_clip = np.clip(data_risk, 0, q99)

        bp = ax.boxplot([data_benign_clip, data_risk_clip],
                        labels=["Benigno", "Riesgo"],
                        patch_artist=True,
                        widths=0.5)
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        ax.set_title(col, fontsize=7)
        ax.tick_params(labelsize=7)

    plt.suptitle("Perfil de comportamiento normal vs anomalo\n"
                 "(Features con mayor varianza en clase benigna)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "05_perfil_comportamiento_normal.png")
    plt.close(fig)


def run_eda(df, feature_cols):
    """Ejecuta todo el analisis exploratorio y genera las graficas."""
    print("\n--- Ejecutando analisis exploratorio de datos ---")
    plot_class_distribution(df)
    print("  [OK] Distribucion de clases")

    top_features = plot_feature_distributions(df, feature_cols)
    print("  [OK] Distribuciones de features top")

    plot_temporal_patterns(df)
    print("  [OK] Patrones temporales")

    plot_correlation_heatmap(df, feature_cols)
    print("  [OK] Heatmap de correlacion")

    plot_normal_behavior_profile(df, feature_cols)
    print("  [OK] Perfil de comportamiento normal")

    print(f"  Figuras guardadas en: {config.FIGURES_DIR}")
    return top_features
