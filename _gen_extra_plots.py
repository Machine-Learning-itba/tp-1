"""Genera graficos extra del notebook como PNG para la presentacion."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})

from preprocessing import load_data

X_train, X_test, y_train, y_test = load_data()
df_clean = X_train.copy()
df_clean["quality"] = y_train

# 1. Heatmap de correlacion
corr = df_clean.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlacion de Pearson"}, ax=ax)
ax.set_title("Matriz de correlacion", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig("results/heatmap_correlacion.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Distribucion de quality
counts = df_clean["quality"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(counts.index.astype(str), counts.values, color="#6C3483", alpha=0.7, edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            f"{val}\n({100*val/len(df_clean):.1f}%)", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Quality", fontsize=12)
ax.set_ylabel("Cantidad", fontsize=12)
ax.set_title("Distribucion de la variable target (quality)", fontsize=13, fontweight="bold")
ax.set_ylim(0, counts.max() * 1.25)
plt.tight_layout()
fig.savefig("results/distribucion_quality.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Tipo de vino
wine_counts = df_clean["red_wine_type"].value_counts().sort_index()
labels = ["Blanco", "Tinto"]
colors = ["#2980B9", "#C0392B"]
fig, ax = plt.subplots(figsize=(5, 4))
b = ax.bar(labels, wine_counts.values, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(b, wine_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("Cantidad de muestras", fontsize=12)
ax.set_title("Distribucion por tipo de vino", fontsize=13, fontweight="bold")
ax.set_ylim(0, wine_counts.max() * 1.15)
plt.tight_layout()
fig.savefig("results/tipo_vino.png", dpi=150, bbox_inches="tight")
plt.close()

print("OK: heatmap_correlacion.png, distribucion_quality.png, tipo_vino.png")
