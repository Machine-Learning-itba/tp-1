"""
Regresion polinomica + Lasso (L1) con k-fold CV estratificado.
Evalua distintos grados del polinomio y valores de alpha.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

from preprocessing import load_data, get_cv

RESULTS_DIR = "results"

# ── Carga ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = load_data()
skf = get_cv()
features = X_train.columns.tolist()

DEGREES = [1, 2, 3]
ALPHAS = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def evaluate(degree, alpha):
    """Evalua una combinacion (grado, alpha) con k-fold CV."""
    rmse_trains, rmse_vals, r2_trains, r2_vals = [], [], [], []
    n_coefs_nonzero = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_ft, y_ft = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fv, y_fv = X_train.iloc[val_idx], y_train.iloc[val_idx]

        steps = [("scaler", StandardScaler())]
        if degree > 1:
            steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
        if alpha > 0:
            steps.append(("model", Lasso(alpha=alpha, max_iter=50000)))
        else:
            steps.append(("model", LinearRegression()))

        pipe = Pipeline(steps)
        pipe.fit(X_ft, y_ft)

        pred_train = pipe.predict(X_ft)
        pred_val = pipe.predict(X_fv)

        rmse_trains.append(root_mean_squared_error(y_ft, pred_train))
        rmse_vals.append(root_mean_squared_error(y_fv, pred_val))
        r2_trains.append(r2_score(y_ft, pred_train))
        r2_vals.append(r2_score(y_fv, pred_val))

        coefs = pipe.named_steps["model"].coef_
        n_coefs_nonzero.append(np.sum(np.abs(coefs) > 1e-6))

    return {
        "degree": degree,
        "alpha": alpha,
        "rmse_train": np.mean(rmse_trains),
        "rmse_val": np.mean(rmse_vals),
        "rmse_val_std": np.std(rmse_vals),
        "r2_train": np.mean(r2_trains),
        "r2_val": np.mean(r2_vals),
        "n_coefs_total": len(pipe.named_steps["model"].coef_),
        "n_coefs_nonzero": int(np.mean(n_coefs_nonzero)),
    }


# ── Evaluacion de todas las combinaciones ──────────────────────────────────────
print("=" * 85)
print("REGRESION POLINOMICA + LASSO (L1)")
print("=" * 85)
print()

# Tambien evaluar polinomio sin regularizacion (OLS)
all_results = []

for degree in DEGREES:
    # Primero sin regularizacion
    res = evaluate(degree, alpha=0)
    res["alpha"] = 0
    all_results.append(res)
    print(f"  Grado {degree} | OLS (sin reg.)  | RMSE val={res['rmse_val']:.4f}  train={res['rmse_train']:.4f}  "
          f"R2 val={res['r2_val']:.4f}  coefs={res['n_coefs_total']}")

    # Luego con Lasso para distintos alphas
    for alpha in ALPHAS:
        res = evaluate(degree, alpha)
        all_results.append(res)
        print(f"  Grado {degree} | alpha={alpha:<8} | RMSE val={res['rmse_val']:.4f}  train={res['rmse_train']:.4f}  "
              f"R2 val={res['r2_val']:.4f}  coefs={res['n_coefs_nonzero']}/{res['n_coefs_total']}")

    print()

df_all = pd.DataFrame(all_results)

# ── Mejor modelo ───────────────────────────────────────────────────────────────
best = df_all.loc[df_all["rmse_val"].idxmin()]

print(f"{'─' * 85}")
print(f"MEJOR MODELO:")
print(f"  Grado:     {int(best['degree'])}")
print(f"  Alpha:     {best['alpha']}")
print(f"  RMSE val:  {best['rmse_val']:.4f} +/- {best['rmse_val_std']:.4f}")
print(f"  RMSE train:{best['rmse_train']:.4f}")
print(f"  R2 val:    {best['r2_val']:.4f}")
print(f"  R2 train:  {best['r2_train']:.4f}")
print(f"  Coefs no-zero: {int(best['n_coefs_nonzero'])}/{int(best['n_coefs_total'])}")

gap = best['rmse_val'] - best['rmse_train']
print(f"  Gap train-val: {gap:.4f} ", end="")
if gap < 0.05:
    print("(bajo)")
elif gap < 0.15:
    print("(moderado)")
else:
    print("(alto, posible overfitting)")

# ── Comparacion por grado ──────────────────────────────────────────────────────
print(f"\n{'─' * 85}")
print("MEJOR ALPHA POR GRADO:\n")
for degree in DEGREES:
    subset = df_all[df_all["degree"] == degree]
    best_for_deg = subset.loc[subset["rmse_val"].idxmin()]
    ols = subset[subset["alpha"] == 0].iloc[0]
    improvement = ols["rmse_val"] - best_for_deg["rmse_val"]
    print(f"  Grado {degree}: alpha={best_for_deg['alpha']:<8}  "
          f"RMSE val={best_for_deg['rmse_val']:.4f}  "
          f"(OLS={ols['rmse_val']:.4f}, mejora Lasso={improvement:+.4f})")

# ── Coeficientes del mejor modelo ─────────────────────────────────────────────
print(f"\n{'─' * 85}")
print(f"COEFICIENTES NO-ZERO del mejor modelo (grado={int(best['degree'])}, alpha={best['alpha']}):\n")

steps = [("scaler", StandardScaler())]
if int(best["degree"]) > 1:
    steps.append(("poly", PolynomialFeatures(degree=int(best["degree"]), include_bias=False)))
if best["alpha"] > 0:
    steps.append(("model", Lasso(alpha=best["alpha"], max_iter=50000)))
else:
    steps.append(("model", LinearRegression()))

best_pipe = Pipeline(steps)
best_pipe.fit(X_train, y_train)

coefs = best_pipe.named_steps["model"].coef_
if "poly" in best_pipe.named_steps:
    feat_names = best_pipe.named_steps["poly"].get_feature_names_out(features)
else:
    feat_names = features

coef_series = pd.Series(coefs, index=feat_names)
nonzero = coef_series[coef_series.abs() > 1e-6].sort_values(key=abs, ascending=False)

for name, val in nonzero.head(20).items():
    print(f"  {name:40s}  {val:+.4f}")
if len(nonzero) > 20:
    print(f"  ... y {len(nonzero) - 20} mas")
print(f"\n  Total: {len(nonzero)} no-zero de {len(coefs)} coeficientes "
      f"({len(coefs) - len(nonzero)} eliminados por Lasso)")

# ── Graficos ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
fig.suptitle("Regresion polinomica + Lasso", fontsize=13, fontweight="bold")

# 1. RMSE vs Alpha por grado
ax = axes[0]
colors_deg = {1: "#2980B9", 2: "#C0392B", 3: "#27AE60"}
for degree in DEGREES:
    subset = df_all[(df_all["degree"] == degree) & (df_all["alpha"] > 0)]
    ax.semilogx(subset["alpha"], subset["rmse_val"], "o-", color=colors_deg[degree],
                label=f"Grado {degree} (val)", markersize=5)
    ax.semilogx(subset["alpha"], subset["rmse_train"], "o--", color=colors_deg[degree],
                alpha=0.4, label=f"Grado {degree} (train)", markersize=3)
ax.set_xlabel("Alpha (log)")
ax.set_ylabel("RMSE")
ax.set_title("RMSE vs Alpha")
ax.legend(fontsize=7)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# 2. Coefs no-zero vs Alpha
ax = axes[1]
for degree in DEGREES:
    subset = df_all[(df_all["degree"] == degree) & (df_all["alpha"] > 0)]
    ax.semilogx(subset["alpha"], subset["n_coefs_nonzero"], "o-", color=colors_deg[degree],
                label=f"Grado {degree} (total={subset['n_coefs_total'].iloc[0]})", markersize=5)
ax.set_xlabel("Alpha (log)")
ax.set_ylabel("Coeficientes no-zero")
ax.set_title("Feature selection por Lasso")
ax.legend(fontsize=7)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# 3. Heatmap RMSE val
ax = axes[2]
lasso_results = df_all[df_all["alpha"] > 0].copy()
pivot = lasso_results.pivot_table(index="degree", columns="alpha", values="rmse_val")
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax,
            cbar_kws={"label": "RMSE val"})
ax.set_title("RMSE validacion (grado x alpha)")
ax.set_ylabel("Grado polinomio")
ax.set_xlabel("Alpha")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/regresion_lasso.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nGrafico guardado: {RESULTS_DIR}/regresion_lasso.png")
