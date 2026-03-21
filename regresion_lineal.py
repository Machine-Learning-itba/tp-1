"""
Regresion lineal con todas las features + k-fold CV estratificado.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

from preprocessing import load_data, get_cv

RESULTS_DIR = "results"

# ── Carga ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = load_data()
skf = get_cv()
features = X_train.columns.tolist()

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# ── K-Fold CV ──────────────────────────────────────────────────────────────────
results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    pipe.fit(X_fold_train, y_fold_train)

    y_pred_train = pipe.predict(X_fold_train)
    y_pred_val = pipe.predict(X_fold_val)

    results.append({
        "fold": fold,
        "rmse_train": root_mean_squared_error(y_fold_train, y_pred_train),
        "rmse_val":   root_mean_squared_error(y_fold_val, y_pred_val),
        "r2_train":   r2_score(y_fold_train, y_pred_train),
        "r2_val":     r2_score(y_fold_val, y_pred_val),
    })

df_results = pd.DataFrame(results)

# ── Resultados por fold ────────────────────────────────────────────────────────
print("=" * 65)
print("REGRESION LINEAL — TODAS LAS FEATURES")
print("=" * 65)
print(f"\nFeatures: {len(features)}")
print(f"Train size: {len(X_train)} | Folds: {skf.n_splits}\n")

print(df_results.to_string(index=False, float_format="%.4f"))

print(f"\n{'─' * 65}")
print(f"RMSE train:  {df_results['rmse_train'].mean():.4f} +/- {df_results['rmse_train'].std():.4f}")
print(f"RMSE val:    {df_results['rmse_val'].mean():.4f} +/- {df_results['rmse_val'].std():.4f}")
print(f"R2 train:    {df_results['r2_train'].mean():.4f} +/- {df_results['r2_train'].std():.4f}")
print(f"R2 val:      {df_results['r2_val'].mean():.4f} +/- {df_results['r2_val'].std():.4f}")

gap = df_results["rmse_val"].mean() - df_results["rmse_train"].mean()
print(f"\nGap train-val (RMSE): {gap:.4f} ", end="")
print("(bajo, no hay overfitting)" if gap < 0.05 else "(posible overfitting)")

# ── Modelo final sobre todo el train para inspeccionar coeficientes ────────────
pipe.fit(X_train, y_train)
coefs = pd.Series(pipe.named_steps["model"].coef_, index=features)
intercept = pipe.named_steps["model"].intercept_

print(f"\n{'─' * 65}")
print(f"Coeficientes (modelo entrenado sobre todo el train, features escaladas):\n")
coefs_sorted = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
for name, val in coefs_sorted.items():
    bar = "+" * int(abs(val) * 20) if val > 0 else "-" * int(abs(val) * 20)
    print(f"  {name:25s}  {val:+.4f}  {bar}")
print(f"  {'intercept':25s}  {intercept:+.4f}")

# ── Graficos ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Regresion lineal — Todas las features", fontsize=13, fontweight="bold")

# 1. Coeficientes
ax = axes[0]
colors = ["#C0392B" if v < 0 else "#2980B9" for v in coefs_sorted.values]
ax.barh(coefs_sorted.index, coefs_sorted.values, color=colors, edgecolor="white")
ax.set_xlabel("Coeficiente (estandarizado)")
ax.set_title("Coeficientes del modelo")
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.5)

# 2. Predicted vs Actual
ax = axes[1]
y_pred_all = pipe.predict(X_train)
ax.scatter(y_train, y_pred_all, alpha=0.15, s=10, color="#6C3483")
lims = [y_train.min() - 0.5, y_train.max() + 0.5]
ax.plot(lims, lims, "--", color="black", linewidth=1, label="Ideal")
ax.set_xlabel("Calidad real")
ax.set_ylabel("Calidad predicha")
ax.set_title("Predicho vs Real (train)")
ax.legend()

# 3. Distribucion de residuos
ax = axes[2]
residuals = y_train.values - y_pred_all
ax.hist(residuals, bins=50, color="#6C3483", alpha=0.6, edgecolor="white", density=True)
ax.set_xlabel("Residuo (real - predicho)")
ax.set_ylabel("Densidad")
ax.set_title(f"Residuos (media={residuals.mean():.3f}, std={residuals.std():.3f})")
ax.axvline(0, color="black", linewidth=1, linestyle="--")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/regresion_lineal.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nGrafico guardado: {RESULTS_DIR}/regresion_lineal.png")
