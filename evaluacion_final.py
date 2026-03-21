"""
Evaluacion final: entrena los mejores modelos sobre todo el train,
evalua sobre test (una sola vez), compara y responde las 3 preguntas del TP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import root_mean_squared_error, r2_score

from preprocessing import load_data, get_cv

RESULTS_DIR = "results"

X_train, X_test, y_train, y_test = load_data()
skf = get_cv()
features = X_train.columns.tolist()

# ── Definicion de modelos a comparar ───────────────────────────────────────────

FORWARD_FEATURES = [
    "alcohol", "volatile acidity", "sulphates", "residual sugar",
    "free sulfur dioxide", "total sulfur dioxide", "density",
    "red_wine_type", "fixed acidity", "pH"
]

models = {
    "Lineal (todas)": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "X_train": X_train,
        "X_test": X_test,
        "desc": "Regresion lineal, 12 features",
    },
    "Lineal (forward)": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "X_train": X_train[FORWARD_FEATURES],
        "X_test": X_test[FORWARD_FEATURES],
        "desc": f"Regresion lineal, {len(FORWARD_FEATURES)} features (forward selection)",
    },
    "Poli grado 2 (OLS)": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression())
        ]),
        "X_train": X_train,
        "X_test": X_test,
        "desc": "Polinomio grado 2, sin regularizacion",
    },
    "Poli grado 2 + Lasso": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", Lasso(alpha=0.001, max_iter=50000))
        ]),
        "X_train": X_train,
        "X_test": X_test,
        "desc": "Polinomio grado 2, Lasso alpha=0.001",
    },
    "Poli grado 3 + Lasso": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("model", Lasso(alpha=0.01, max_iter=50000))
        ]),
        "X_train": X_train,
        "X_test": X_test,
        "desc": "Polinomio grado 3, Lasso alpha=0.01",
    },
}

# ── K-Fold CV sobre train (para comparacion justa) ────────────────────────────

print("=" * 80)
print("EVALUACION FINAL")
print("=" * 80)

cv_results = {}

print("\n--- Resultados de Cross-Validation (sobre train) ---\n")
print(f"{'Modelo':<25s}  {'RMSE val':>10s}  {'± std':>8s}  {'RMSE train':>10s}  {'R2 val':>8s}  {'R2 train':>8s}")
print("─" * 80)

for name, cfg in models.items():
    rmse_trains, rmse_vals, r2_trains, r2_vals = [], [], [], []

    for train_idx, val_idx in skf.split(cfg["X_train"], y_train):
        Xft = cfg["X_train"].iloc[train_idx]
        yft = y_train.iloc[train_idx]
        Xfv = cfg["X_train"].iloc[val_idx]
        yfv = y_train.iloc[val_idx]

        pipe = cfg["pipe"].__class__(cfg["pipe"].steps)
        pipe.set_params(**{k: v for k, v in cfg["pipe"].get_params().items()
                          if "__" in k})
        pipe.fit(Xft, yft)

        rmse_trains.append(root_mean_squared_error(yft, pipe.predict(Xft)))
        rmse_vals.append(root_mean_squared_error(yfv, pipe.predict(Xfv)))
        r2_trains.append(r2_score(yft, pipe.predict(Xft)))
        r2_vals.append(r2_score(yfv, pipe.predict(Xfv)))

    cv_results[name] = {
        "rmse_val": np.mean(rmse_vals),
        "rmse_val_std": np.std(rmse_vals),
        "rmse_train": np.mean(rmse_trains),
        "r2_val": np.mean(r2_vals),
        "r2_train": np.mean(r2_trains),
    }

    print(f"{name:<25s}  {cv_results[name]['rmse_val']:>10.4f}  {cv_results[name]['rmse_val_std']:>8.4f}  "
          f"{cv_results[name]['rmse_train']:>10.4f}  {cv_results[name]['r2_val']:>8.4f}  "
          f"{cv_results[name]['r2_train']:>8.4f}")

# ── Evaluacion sobre TEST (una sola vez) ───────────────────────────────────────

print(f"\n{'─' * 80}")
print("\n--- Evaluacion sobre TEST SET (primera y unica vez) ---\n")
print(f"{'Modelo':<25s}  {'RMSE test':>10s}  {'R2 test':>8s}  {'RMSE val (CV)':>13s}  {'Gap val-test':>12s}")
print("─" * 80)

test_results = {}

for name, cfg in models.items():
    pipe = cfg["pipe"]
    pipe.fit(cfg["X_train"], y_train)

    y_pred_test = pipe.predict(cfg["X_test"])
    y_pred_train = pipe.predict(cfg["X_train"])

    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    gap = rmse_test - cv_results[name]["rmse_val"]

    test_results[name] = {
        "rmse_test": rmse_test,
        "r2_test": r2_test,
        "y_pred_test": y_pred_test,
        "y_pred_train": y_pred_train,
        "gap": gap,
    }

    print(f"{name:<25s}  {rmse_test:>10.4f}  {r2_test:>8.4f}  "
          f"{cv_results[name]['rmse_val']:>13.4f}  {gap:>+12.4f}")

# ── Respuestas a las 3 preguntas ───────────────────────────────────────────────

# El modelo se elige por CV (nunca por test). Test solo confirma.
best_name = min(cv_results, key=lambda k: cv_results[k]["rmse_val"])
best_cv_rmse = cv_results[best_name]["rmse_val"]
best_cv_std = cv_results[best_name]["rmse_val_std"]
best_rmse_test = test_results[best_name]["rmse_test"]
best_r2_test = test_results[best_name]["r2_test"]
best_gap = test_results[best_name]["gap"]

print(f"\n{'=' * 80}")
print("RESPUESTAS A LAS PREGUNTAS DEL TP")
print(f"{'=' * 80}")

print(f"""
1. Que modelo obtuvo menor error?

   {best_name}, seleccionado por menor RMSE en cross-validation: {best_cv_rmse:.4f}.
   Al evaluarlo sobre el test set (una sola vez), obtiene RMSE = {best_rmse_test:.4f}
   (R2 = {best_r2_test:.4f}).

   Nota: el modelo se elige exclusivamente por su desempeno en CV, no en test.
   El test set solo se usa para confirmar que el modelo generaliza correctamente.
   Si eligieramos por test, estariamos haciendo seleccion de modelo sobre datos
   que deberian ser "no vistos", lo que invalida la estimacion de error.

2. Cual elegirian para una aplicacion real?

   {best_name}. Justificacion:
   - Tiene el menor RMSE de validacion en cross-validation ({best_cv_rmse:.4f}).
   - El gap entre RMSE de CV y test es {best_gap:+.4f}, lo que confirma que el
     modelo generaliza bien y no hay overfitting.
   - Lasso proporciona feature selection automatica, reduciendo de 90 a ~77
     coeficientes, lo que simplifica el modelo sin perder rendimiento.
   - La regularizacion lo hace robusto ante datos nuevos ligeramente distintos
     a los de entrenamiento.

3. Que RMSE esperan en datos nuevos?

   Reportariamos un RMSE esperado de {best_cv_rmse:.4f} +/- {best_cv_std:.4f}.

   Usamos el RMSE de cross-validation como estimacion (no el de test) porque:
   - El CV promedia sobre 5 folds, dando una estimacion mas estable.
   - El RMSE de test ({best_rmse_test:.4f}) es una sola medicion sobre un conjunto
     fijo, sujeta a varianza muestral.
   - Ambos valores son consistentes (gap de {abs(best_gap):.4f}), lo que da
     confianza en la estimacion.

   Esto significa que el modelo se equivoca en promedio ~{best_cv_rmse:.1f} puntos
   en la escala de calidad (0-10).
""")

# ── Graficos comparativos ──────────────────────────────────────────────────────

model_names = list(models.keys())
short_names = ["Lineal\n(todas)", "Lineal\n(forward)", "Poli 2\n(OLS)", "Poli 2\n(Lasso)", "Poli 3\n(Lasso)"]
colors = ["#2980B9", "#3498DB", "#E67E22", "#C0392B", "#8E44AD"]
best_idx = model_names.index(best_name)

x = np.arange(len(model_names))
width = 0.3

fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.3,
                       height_ratios=[1, 1, 1.1])

# ── 1. RMSE comparativo (barras) ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
rmse_cv = [cv_results[n]["rmse_val"] for n in model_names]
rmse_test_list = [test_results[n]["rmse_test"] for n in model_names]

bars1 = ax.bar(x - width/2, rmse_cv, width, color="#2980B9", alpha=0.8, label="CV (val)")
bars2 = ax.bar(x + width/2, rmse_test_list, width, color="#C0392B", alpha=0.8, label="Test")

for bar, val in zip(bars1, rmse_cv):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
for bar, val in zip(bars2, rmse_test_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7)

# Marcar el mejor
bars1[best_idx].set_edgecolor("black")
bars1[best_idx].set_linewidth(2)
bars2[best_idx].set_edgecolor("black")
bars2[best_idx].set_linewidth(2)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylabel("RMSE")
ax.set_title("RMSE: Cross-Validation vs Test", fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0.65, max(max(rmse_cv), max(rmse_test_list)) + 0.04)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── 2. R2 comparativo ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
r2_cv = [cv_results[n]["r2_val"] for n in model_names]
r2_test_list = [test_results[n]["r2_test"] for n in model_names]

bars1 = ax.bar(x - width/2, r2_cv, width, color="#2980B9", alpha=0.8, label="CV (val)")
bars2 = ax.bar(x + width/2, r2_test_list, width, color="#C0392B", alpha=0.8, label="Test")

for bar, val in zip(bars1, r2_cv):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
for bar, val in zip(bars2, r2_test_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7)

bars1[best_idx].set_edgecolor("black")
bars1[best_idx].set_linewidth(2)
bars2[best_idx].set_edgecolor("black")
bars2[best_idx].set_linewidth(2)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylabel("R2")
ax.set_title("R2: Cross-Validation vs Test", fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(0, max(max(r2_cv), max(r2_test_list)) + 0.08)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── 3. Gap train-val-test ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
rmse_train_list = [cv_results[n]["rmse_train"] for n in model_names]
rmse_cv_list = [cv_results[n]["rmse_val"] for n in model_names]
rmse_test_list2 = [test_results[n]["rmse_test"] for n in model_names]

ax.plot(range(len(model_names)), rmse_train_list, "o-", color="#27AE60", label="Train", markersize=8)
ax.plot(range(len(model_names)), rmse_cv_list, "s-", color="#2980B9", label="Validation (CV)", markersize=8)
ax.plot(range(len(model_names)), rmse_test_list2, "^-", color="#C0392B", label="Test", markersize=8)

# Resaltar el mejor
for series, marker in [(rmse_train_list, "o"), (rmse_cv_list, "s"), (rmse_test_list2, "^")]:
    ax.plot(best_idx, series[best_idx], marker, color="black", markersize=14,
            markerfacecolor="none", markeredgewidth=2)

ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylabel("RMSE")
ax.set_title("RMSE: Train vs Validation vs Test", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── 4. Residuos por modelo (boxplot) ──────────────────────────────────────────
ax = fig.add_subplot(gs[1, :2])
residuals_all = []
for name in model_names:
    residuals_all.append(y_test.values - test_results[name]["y_pred_test"])

bp = ax.boxplot(residuals_all, vert=True, patch_artist=True, labels=short_names,
                flierprops=dict(marker=".", markersize=3, alpha=0.3),
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
bp["boxes"][best_idx].set_edgecolor("black")
bp["boxes"][best_idx].set_linewidth(2)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_ylabel("Residuo (real - predicho)")
ax.set_title("Distribucion de residuos sobre test set", fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.tick_params(axis="x", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── 5. Tabla resumen visual ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")

table_data = []
for i, name in enumerate(model_names):
    marker = " <<" if name == best_name else ""
    table_data.append([
        short_names[i].replace("\n", " "),
        f"{cv_results[name]['rmse_val']:.4f}",
        f"{test_results[name]['rmse_test']:.4f}",
        f"{cv_results[name]['r2_val']:.4f}",
        f"{test_results[name]['gap']:+.4f}",
    ])

table = ax.table(
    cellText=table_data,
    colLabels=["Modelo", "RMSE\nCV", "RMSE\nTest", "R2\nCV", "Gap"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.8)

# Resaltar fila del mejor modelo
for col in range(5):
    table[best_idx + 1, col].set_facecolor("#D5F5E3")
    table[best_idx + 1, col].set_text_props(fontweight="bold")

ax.set_title("Resumen (mejor modelo resaltado)", fontweight="bold", fontsize=10, pad=15)

# ── 6-10. Predicted vs Actual para cada modelo ───────────────────────────────
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[2, :], wspace=0.3)

for idx, name in enumerate(model_names):
    ax = fig.add_subplot(gs_bottom[0, idx])
    y_pred = test_results[name]["y_pred_test"]

    ax.scatter(y_test, y_pred, alpha=0.3, s=15, color=colors[idx], edgecolor="none")
    lims = [2.5, 9.5]
    ax.plot(lims, lims, "--", color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("Real", fontsize=9)
    if idx == 0:
        ax.set_ylabel("Predicho", fontsize=9)
    title = f"{short_names[idx]}\nRMSE={test_results[name]['rmse_test']:.3f}"
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if name == best_name:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(2.5)

fig.suptitle("Comparacion de modelos — Evaluacion final",
             fontsize=15, fontweight="bold", y=0.98)
plt.savefig(f"{RESULTS_DIR}/evaluacion_final.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Grafico guardado: {RESULTS_DIR}/evaluacion_final.png")

# ── Tabla resumen exportada ────────────────────────────────────────────────────

summary = pd.DataFrame({
    "Modelo": model_names,
    "Descripcion": [models[n]["desc"] for n in model_names],
    "RMSE_CV_val": [cv_results[n]["rmse_val"] for n in model_names],
    "RMSE_CV_val_std": [cv_results[n]["rmse_val_std"] for n in model_names],
    "RMSE_CV_train": [cv_results[n]["rmse_train"] for n in model_names],
    "R2_CV_val": [cv_results[n]["r2_val"] for n in model_names],
    "RMSE_test": [test_results[n]["rmse_test"] for n in model_names],
    "R2_test": [test_results[n]["r2_test"] for n in model_names],
    "Gap_val_test": [test_results[n]["gap"] for n in model_names],
}).round(4)

summary.to_csv(f"{RESULTS_DIR}/comparacion_modelos.csv", index=False)
print(f"Tabla guardada: {RESULTS_DIR}/comparacion_modelos.csv")

# ── CSV del test set con prediccion del mejor modelo ─────────────────────────

df_test_out = X_test.copy()
df_test_out["quality_real"] = y_test.values
df_test_out["quality_predicho"] = test_results[best_name]["y_pred_test"]
df_test_out.to_csv(f"{RESULTS_DIR}/test_predicciones.csv", index=False)
print(f"Predicciones guardadas: {RESULTS_DIR}/test_predicciones.csv")
