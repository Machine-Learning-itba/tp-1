"""
Forward Selection con regresion lineal + k-fold CV estratificado.
En cada paso agrega la feature que mas reduce el RMSE de validacion.
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
all_features = X_train.columns.tolist()


def evaluate_features(feature_list):
    """Evalua un subconjunto de features con k-fold CV. Retorna RMSE train/val."""
    rmse_trains, rmse_vals = [], []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_ft = X_train[feature_list].iloc[train_idx]
        y_ft = y_train.iloc[train_idx]
        X_fv = X_train[feature_list].iloc[val_idx]
        y_fv = y_train.iloc[val_idx]

        pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        pipe.fit(X_ft, y_ft)

        rmse_trains.append(root_mean_squared_error(y_ft, pipe.predict(X_ft)))
        rmse_vals.append(root_mean_squared_error(y_fv, pipe.predict(X_fv)))

    return np.mean(rmse_trains), np.mean(rmse_vals)


# ── Forward Selection ──────────────────────────────────────────────────────────
selected = []
remaining = list(all_features)
history = []

print("=" * 75)
print("FORWARD SELECTION — REGRESION LINEAL")
print("=" * 75)
print()

for step in range(1, len(all_features) + 1):
    best_feat = None
    best_rmse_val = np.inf
    best_rmse_train = np.inf

    for feat in remaining:
        candidate = selected + [feat]
        rmse_tr, rmse_val = evaluate_features(candidate)
        if rmse_val < best_rmse_val:
            best_feat = feat
            best_rmse_val = rmse_val
            best_rmse_train = rmse_tr

    selected.append(best_feat)
    remaining.remove(best_feat)
    history.append({
        "step": step,
        "feature_added": best_feat,
        "rmse_train": best_rmse_train,
        "rmse_val": best_rmse_val,
    })

    print(f"  Paso {step:2d}: +{best_feat:25s}  RMSE val={best_rmse_val:.4f}  train={best_rmse_train:.4f}")

df_history = pd.DataFrame(history)

# ── Encontrar el punto optimo ──────────────────────────────────────────────────
# El punto optimo es donde agregar una feature mas no mejora el RMSE de validacion
best_idx = df_history["rmse_val"].idxmin()
optimal_n = best_idx + 1
optimal_features = selected[:optimal_n]

print(f"\n{'─' * 75}")
print(f"Punto optimo: {optimal_n} features")
print(f"RMSE val optimo: {df_history.loc[best_idx, 'rmse_val']:.4f}")
print(f"Features seleccionadas: {optimal_features}")

# Comparar con usar todas
all_rmse_tr, all_rmse_val = evaluate_features(all_features)
opt_rmse_tr, opt_rmse_val = evaluate_features(optimal_features)

print(f"\n{'─' * 75}")
print(f"Comparacion:")
print(f"  Todas las features ({len(all_features):2d}):  RMSE val = {all_rmse_val:.4f}  train = {all_rmse_tr:.4f}")
print(f"  Forward selection  ({optimal_n:2d}):  RMSE val = {opt_rmse_val:.4f}  train = {opt_rmse_tr:.4f}")
diff = all_rmse_val - opt_rmse_val
print(f"  Diferencia: {diff:+.4f} ", end="")
if abs(diff) < 0.005:
    print("(despreciable, las features extra no aportan ni perjudican)")
elif diff > 0:
    print("(el subconjunto es mejor, hay features que agregan ruido)")
else:
    print("(todas las features es mejor)")

# Features descartadas
discarded = [f for f in all_features if f not in optimal_features]
if discarded:
    print(f"\n  Features descartadas: {discarded}")

# ── R2 del modelo optimo ───────────────────────────────────────────────────────
r2_trains, r2_vals = [], []
for train_idx, val_idx in skf.split(X_train, y_train):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    pipe.fit(X_train[optimal_features].iloc[train_idx], y_train.iloc[train_idx])
    r2_trains.append(r2_score(y_train.iloc[train_idx], pipe.predict(X_train[optimal_features].iloc[train_idx])))
    r2_vals.append(r2_score(y_train.iloc[val_idx], pipe.predict(X_train[optimal_features].iloc[val_idx])))

print(f"\n  R2 val (modelo optimo):  {np.mean(r2_vals):.4f} +/- {np.std(r2_vals):.4f}")
print(f"  R2 train (modelo optimo): {np.mean(r2_trains):.4f} +/- {np.std(r2_trains):.4f}")

# ── Graficos ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Forward Selection — Regresion lineal", fontsize=13, fontweight="bold")

# 1. RMSE vs numero de features
ax = axes[0]
ax.plot(df_history["step"], df_history["rmse_train"], "o-", color="#2980B9", label="Train", markersize=5)
ax.plot(df_history["step"], df_history["rmse_val"], "o-", color="#C0392B", label="Validacion", markersize=5)
ax.axvline(optimal_n, color="gray", linestyle="--", alpha=0.7, label=f"Optimo ({optimal_n} features)")
ax.set_xlabel("Numero de features")
ax.set_ylabel("RMSE (media k-fold)")
ax.set_title("RMSE vs cantidad de features")
ax.set_xticks(df_history["step"])
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.3)

# 2. Orden de seleccion
ax = axes[1]
colors = ["#2980B9" if i < optimal_n else "#BDC3C7" for i in range(len(selected))]
ax.barh(range(len(selected)), df_history["rmse_val"], color=colors, edgecolor="white")
ax.set_yticks(range(len(selected)))
ax.set_yticklabels([f"{i+1}. {f}" for i, f in enumerate(selected)], fontsize=8)
ax.set_xlabel("RMSE validacion (acumulado)")
ax.set_title("Orden de seleccion (gris = descartable)")
ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/forward_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nGrafico guardado: {RESULTS_DIR}/forward_selection.png")
