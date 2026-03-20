# Estrategia de limpieza y preprocesamiento del dataset Wine Quality

**Fecha:** 2026-03-20
**Dataset:** Wine Quality (Cortez et al., 2009) — Vinho Verde, Portugal
**Registros totales:** 6497 (1599 tintos + 4898 blancos)
**Variables:** 11 fisicoquimicas (input) + 1 sensorial (output: quality)

---

## 1. Codificacion de la variable categorica `wine_type`

El dataset se compone de dos archivos separados (tinto y blanco). Al unirlos se agrega
la columna `wine_type` como variable categorica binaria.

**Metodo:** One-Hot Encoding con `drop_first=True`.

- Se genera una unica columna `is_white` (0 = tinto, 1 = blanco).
- Se descarta la segunda columna (`is_red`) porque es linealmente dependiente:
  `is_red = 1 - is_white`. Mantener ambas introduce multicolinealidad perfecta,
  lo cual perjudica a modelos lineales (regresion logistica, SVM lineal).

**Justificacion:** Las distribuciones de varias variables difieren significativamente
entre tipos de vino (ej: `total sulfur dioxide` media 46 en tinto vs 138 en blanco,
`volatile acidity` media 0.53 en tinto vs 0.28 en blanco). Incluir el tipo como
feature permite al modelo capturar estas diferencias sistematicas.

---

## 2. Valores erroneos: eliminacion de filas

Los siguientes valores se consideran **fisicamente imposibles, regulatoriamente
ilegales, o indicativos de error de medicion/entrada**. Se eliminan las filas
completas porque no representan vinos reales comercializables y su presencia
distorsionaria el entrenamiento del modelo.

### 2.1 `total sulfur dioxide` > 300 mg/L — 6 filas (todas blanco)

| Indice | Valor   | Tipo  |
|--------|---------|-------|
| 1924   | 313.0   | white |
| 3016   | 366.5   | white |
| 3530   | 307.5   | white |
| 3726   | 344.0   | white |
| 4253   | 303.0   | white |
| 6344   | 440.0   | white |

**Por que se eliminan:**
- La regulacion de la Union Europea (Reg. 2019/934, Anexo I-B) establece limites
  maximos de SO2 total de **150 mg/L para vinos tintos** y **200 mg/L para blancos**
  (con excepciones de hasta 300 mg/L para vinos con azucar residual > 5 g/L).
- Valores por encima de 300 mg/L exceden incluso el limite mas permisivo. En
  particular, el valor de 440 mg/L es mas del doble del limite estandar.
- Concentraciones tan altas son toxicas y no pasarian control de calidad.
  Interpretamos estos valores como errores de medicion del instrumento o errores
  de transcripcion (ej: punto decimal desplazado).

### 2.2 `free sulfur dioxide` > 150 mg/L — 1 fila

| Indice | Valor   | Tipo  |
|--------|---------|-------|
| 6344   | 289.0   | white |

**Por que se elimina:**
- El SO2 libre es una fraccion del SO2 total. En la practica enologica raramente
  supera los 80-100 mg/L, incluso en vinos blancos con alta sulfitacion.
- Un valor de 289 mg/L es fisicamente incoherente con la vinificacion normal.
- Nota: esta fila (6344) tambien viola el criterio de SO2 total (440 mg/L),
  lo que refuerza la hipotesis de error sistematico en esa medicion.

### 2.3 `chlorides` > 0.3 g/L — 24 filas (22 tintos, 2 blancos)

**Por que se eliminan:**
- El contenido de cloruros (como NaCl) en vino tipicamente oscila entre
  0.01 y 0.10 g/L. Valores por encima de 0.2 g/L son perceptibles como
  gusto salado y se consideran un defecto.
- El rango 0.30-0.61 g/L que presentan estas 24 filas es entre 3x y 6x el
  maximo normal. Estos valores sugieren:
  - Contaminacion de la muestra durante la medicion.
  - Error de unidades (mg/L registrado como g/L, o viceversa).
  - Vino con defecto grave que no es representativo de Vinho Verde comercial.
- El valor maximo (0.611 g/L, indice 258, tinto) equivale a una salinidad
  perceptible al paladar, incompatible con un vino comercializable.

### 2.4 `citric acid` > 1.0 g/L — 2 filas (ambas blanco)

| Indice | Valor | Tipo  |
|--------|-------|-------|
| 2344   | 1.66  | white |
| 4751   | 1.23  | white |

**Por que se eliminan:**
- El acido citrico en vino se encuentra naturalmente entre 0 y 0.5 g/L.
  La adicion de acido citrico esta permitida en la UE pero con limites
  que no superan 1.0 g/L en el producto final.
- El valor de 1.66 g/L es 3.3x el maximo natural y excede el limite legal
  de adicion. Probablemente es un error de medicion o transcripcion.

### 2.5 `density` > 1.010 g/cm3 — 3 filas (todas blanco)

| Indice | Densidad | Azucar residual | Tipo  |
|--------|----------|-----------------|-------|
| 3252   | 1.01030  | 31.6 g/L        | white |
| 3262   | 1.01030  | 31.6 g/L        | white |
| 4380   | 1.03898  | 65.8 g/L        | white |

**Por que se eliminan:**
- La densidad del vino de mesa esta tipicamente entre 0.985 y 1.005 g/cm3.
  Densidades superiores a 1.01 corresponden a vinos de postre/licorosos con
  muy alto contenido de azucar.
- El Vinho Verde es una denominacion de origen de **vinos secos y semi-secos**
  (tipicamente < 9 g/L de azucar residual). Los valores de 31.6 y 65.8 g/L
  son incompatibles con esta DO.
- Las filas 3252 y 3262 son sospechosamente identicas (posible duplicado
  de entrada).
- La fila 4380 (densidad 1.039, azucar 65.8 g/L) es extrema incluso para
  vinos de postre. Probablemente es un vino mal clasificado como Vinho Verde
  o un error de transcripcion.

### 2.6 `pH` < 2.80 — 7 filas (1 tinto, 6 blancos)

| Indice | pH   | Tipo  |
|--------|------|-------|
| 151    | 2.74 | red   |
| 2813   | 2.74 | white |
| 3499   | 2.72 | white |
| 3558   | 2.79 | white |
| 3559   | 2.79 | white |
| 3761   | 2.77 | white |
| 5361   | 2.79 | white |

**Por que se eliminan:**
- El pH del vino se encuentra normalmente entre 3.0 y 4.0. Valores
  por debajo de 2.9 son extremadamente raros y corresponden a una acidez
  que haria el vino desagradable e incluso potencialmente irritante.
- Un pH de 2.72 es comparable al del vinagre (pH ~2.4-3.4) o al jugo de
  limon (pH ~2.0-2.6). No es un valor esperable en un vino comercializable.
- Nota: la fila 151 (tinto) tambien viola el criterio de chlorides (0.610 g/L),
  acumulando dos anomalias simultaneas — reforzando la hipotesis de error.

### Resumen de eliminacion

| Criterio                     | Filas afectadas | % del dataset |
|------------------------------|-----------------|---------------|
| total sulfur dioxide > 300   | 6               | 0.09%         |
| free sulfur dioxide > 150    | 1               | 0.02%         |
| chlorides > 0.3              | 24              | 0.37%         |
| citric acid > 1.0            | 2               | 0.03%         |
| density > 1.01               | 3               | 0.05%         |
| pH < 2.80                    | 7               | 0.11%         |
| **Union (sin duplicados)**   | **41**          | **0.63%**     |

**Filas restantes tras limpieza: 6456 (99.4% del dataset original).**

La perdida es minima y no introduce sesgo apreciable en las distribuciones.

---

## 3. Outliers estadisticos: estrategia por variable

Para los valores que son **quimicamente posibles** pero estadisticamente extremos
(outliers por IQR), **no se eliminan filas**. En su lugar se aplica **winsorizacion**
al rango [percentil 1, percentil 99], reemplazando los valores extremos por el
valor del percentil correspondiente.

### Por que winsorizacion y no eliminacion

- **Preserva el tamano muestral:** eliminar outliers en multiples variables
  simultaneamente puede producir una perdida acumulativa significativa
  (una fila puede ser outlier en una variable y perfectamente normal en las demas).
- **Preserva la forma de la distribucion:** solo "aplana" las colas extremas
  sin alterar la estructura central.
- **Es mas robusta que la imputacion por media/mediana:** la media ignora la
  posicion del valor original en la distribucion; la winsorizacion lo ancla al
  extremo mas cercano de la distribucion real.

### Variables a winsorizar

| Variable           | Percentil 1 | Percentil 99 | Filas afectadas | Skewness | Justificacion                                                          |
|--------------------|-------------|--------------|-----------------|----------|------------------------------------------------------------------------|
| `residual sugar`   | 0.90        | 18.20        | 97              | 1.44     | Distribucion bimodal (secos vs semi-dulces). Los extremos son vinos reales, no errores, pero su magnitud distorsiona el escalado. |
| `volatile acidity` | 0.12        | 0.88         | 96              | 1.50     | Valores altos indican acetificacion. Son quimicamente posibles pero extremos para vino comercial.                                 |
| `sulphates`        | 0.30        | 0.99         | 115             | 1.80     | Cola derecha larga. Valores hasta 2.0 son posibles pero muy raros en Vinho Verde.                                                |
| `chlorides`        | *(post-eliminacion, valores restantes <= 0.3)* | | | 5.40 | Tras eliminar los > 0.3, la distribucion aun tiene cola larga. Winsorizar al p1-p99 del subset limpio.                           |
| `citric acid`      | *(post-eliminacion, valores restantes <= 1.0)* | | | 0.47 | Skewness baja tras eliminacion. Evaluar si winsorizar es necesario despues de la limpieza.                                        |

### Variables que NO se winsorizan

| Variable               | Motivo                                                                                                       |
|------------------------|--------------------------------------------------------------------------------------------------------------|
| `fixed acidity`        | Skewness moderada (1.72), pero rango 3.8-15.9 g/L es fisicamente coherente para vino. Distribucion natural.  |
| `free sulfur dioxide`  | Tras eliminar el valor de 289, los restantes estan dentro del rango normal.                                  |
| `total sulfur dioxide` | Tras eliminar los > 300, la distribucion es aproximadamente simetrica (skewness ~0).                         |
| `density`              | Tras eliminar los > 1.01, rango restante 0.987-1.004 es normal para vino.                                   |
| `pH`                   | Tras eliminar los < 2.80, rango 2.80-4.01 es el rango natural del vino.                                     |
| `alcohol`              | Sin outliers estadisticos significativos (0 en blancos, 13 en tintos). Rango 8-14.9% es normal.              |
| `quality`              | **Es la variable target. Nunca se modifican sus valores.** Los vinos de calidad 3 o 9 son raros pero representan los extremos que el clasificador necesita aprender. |

---

## 4. Orden de ejecucion del pipeline de limpieza

```
1. Cargar y unir datasets (red + white)
2. Agregar columna wine_type
3. ELIMINAR filas con valores erroneos (Seccion 2)
4. One-Hot Encoding: wine_type -> is_white (Seccion 1)
5. WINSORIZAR variables con outliers estadisticos al p1-p99 (Seccion 3)
6. Verificar que no haya valores nulos (el dataset original no tiene)
7. Exportar dataset limpio
```

El orden es importante: la eliminacion de filas erroneas se hace **antes** de
calcular los percentiles para la winsorizacion, para que los valores imposibles
no contaminen los limites de corte.

---

## 5. Impacto esperado

- **Filas eliminadas:** 41 (0.63%) — impacto negligible en tamano muestral.
- **Filas winsorizadas:** ~300 valores individuales ajustados (~0.07% de todas las
  celdas del dataset). La estructura de la fila se mantiene intacta.
- **Distribucion post-limpieza:** las colas extremas se reducen, mejorando la
  convergencia de modelos sensibles a escala (SVM, KNN, redes neuronales) y
  reduciendo la influencia desproporcionada de puntos extremos en modelos lineales.
