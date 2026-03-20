"""
Análisis exploratorio del dataset Wine Quality.
Genera para cada variable:
  - Histograma con curva de distribución normal ajustada (campana de Gauss)
  - Boxplot con outliers identificados
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ── Configuración ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "red":      "#C0392B",
    "white":    "#2980B9",
    "combined": "#6C3483",
}

# ── Carga de datos ─────────────────────────────────────────────────────────────
red   = pd.read_csv(os.path.join(DATA_DIR, "winequality-red.csv"),   sep=";")
white = pd.read_csv(os.path.join(DATA_DIR, "winequality-white.csv"), sep=";")

red["wine_type"]   = "red"
white["wine_type"] = "white"

combined = pd.concat([red, white], ignore_index=True)

DATASETS = {
    "red":      red,
    "white":    white,
    "combined": combined,
}

FEATURES = [c for c in red.columns if c != "wine_type"]


# ── Funciones de apoyo ─────────────────────────────────────────────────────────

def plot_distribution(ax, data: pd.Series, color: str, title: str) -> None:
    """Histograma normalizado + curva Gaussiana ajustada (mu, sigma de los datos)."""
    values = data.dropna()

    ax.hist(values, bins=40, density=True, color=color, alpha=0.55,
            edgecolor="white", linewidth=0.4, label="Distribución empírica")

    mu, sigma = values.mean(), values.std()
    x = np.linspace(values.min(), values.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=color, linewidth=2.2,
            label=f"Gauss  μ={mu:.3f}  σ={sigma:.3f}")

    # Test de normalidad Shapiro-Wilk (muestra ≤ 5000)
    sample = values.sample(min(len(values), 5000), random_state=42)
    _, p_value = stats.shapiro(sample)
    normal_text = f"Shapiro-Wilk  p={p_value:.4f}"
    ax.text(0.97, 0.95, normal_text, transform=ax.transAxes, fontsize=7.5,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(data.name, fontsize=8.5)
    ax.set_ylabel("Densidad", fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_boxplot(ax, data: pd.Series, color: str, title: str) -> None:
    """Boxplot horizontal con outliers individuales marcados."""
    values = data.dropna()

    bp = ax.boxplot(values, vert=False, patch_artist=True,
                    flierprops=dict(marker="o", markerfacecolor=color,
                                   markeredgecolor="white", markersize=4,
                                   alpha=0.6),
                    medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(facecolor=color, alpha=0.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.5))

    # Cálculo explícito de outliers (IQR)
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = values[(values < lower) | (values > upper)]
    pct_out = 100 * len(outliers) / len(values)

    info = (f"n={len(values):,}  |  outliers={len(outliers)} ({pct_out:.1f}%)\n"
            f"Q1={q1:.3f}  med={values.median():.3f}  Q3={q3:.3f}  IQR={iqr:.3f}")
    ax.text(0.5, 0.88, info, transform=ax.transAxes, fontsize=7.5,
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(data.name, fontsize=8.5)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


# ── Generación de gráficos ─────────────────────────────────────────────────────

for ds_name, df in DATASETS.items():
    color = COLORS[ds_name]
    n_features = len(FEATURES)

    # ── Figura 1: Distribuciones (campana de Gauss) ───────────────────────────
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.8))
    fig.suptitle(f"Distribución de variables — {ds_name.capitalize()} wine  (n={len(df):,})",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, feature in enumerate(FEATURES):
        ax = axes.flat[idx]
        plot_distribution(ax, df[feature], color, feature)

    # Ocultar ejes sobrantes
    for idx in range(n_features, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"distribucion_{ds_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")

    # ── Figura 2: Boxplots con outliers ───────────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.0))
    fig.suptitle(f"Boxplots con outliers — {ds_name.capitalize()} wine  (n={len(df):,})",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, feature in enumerate(FEATURES):
        ax = axes.flat[idx]
        plot_boxplot(ax, df[feature], color, feature)

    for idx in range(n_features, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"boxplot_{ds_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")

# ── Resumen estadístico en CSV ─────────────────────────────────────────────────
for ds_name, df in DATASETS.items():
    rows = []
    for feature in FEATURES:
        values = df[feature].dropna()
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        outliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
        sample = values.sample(min(len(values), 5000), random_state=42)
        _, p_shapiro = stats.shapiro(sample)
        skewness = values.skew()
        rows.append({
            "variable":        feature,
            "n":               len(values),
            "mean":            round(values.mean(), 4),
            "std":             round(values.std(), 4),
            "min":             round(values.min(), 4),
            "Q1":              round(q1, 4),
            "median":          round(values.median(), 4),
            "Q3":              round(q3, 4),
            "max":             round(values.max(), 4),
            "IQR":             round(iqr, 4),
            "outliers_count":  len(outliers),
            "outliers_%":      round(100 * len(outliers) / len(values), 2),
            "skewness":        round(skewness, 4),
            "shapiro_p":       round(p_shapiro, 6),
            "is_normal(p>0.05)": p_shapiro > 0.05,
        })
    summary = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, f"estadisticas_{ds_name}.csv")
    summary.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}")

print("\nAnalisis completado. Resultados en:", OUT_DIR)
