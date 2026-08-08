# ==============================================================================
# BLOQUE 0 — LIBRERÍAS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from rocauc_comparison import delong_roc_test
import warnings
warnings.filterwarnings('ignore')
 
from ucimlrepo import fetch_ucirepo
 
# Preprocesamiento y partición
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
 
# Modelos
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
 
# Métricas
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
 
# Inferencia estadística para M1
import statsmodels.api as sm
 
# Prueba de DeLong (comparación ROC)
from scipy import stats
 
# Serialización de objetos
import joblib
 
# Paleta visual consistente en todo el script
COLOR_M1   = '#378ADD'   # azul — modelo clásico
COLOR_M2   = '#E24B4A'   # rojo — modelo LASSO
COLOR_NEG  = '#1D9E75'   # verde — clase 0 (no incumple)
COLOR_POS  = '#D4537E'   # rosa  — clase 1 (incumple)
ALPHA      = 0.75
 
 
# ==============================================================================
# BLOQUE 1 — CARGA Y RENOMBRADO DE COLUMNAS
# ==============================================================================
print("=" * 65)
print("BLOQUE 1 — CARGA DE DATOS")
print("=" * 65)
 
# Se descarga directamente del repositorio UCI usando su ID oficial (350)
credit_data = fetch_ucirepo(id=350)
 
X_raw = credit_data.data.features
y_raw = credit_data.data.targets
 
# Unimos predictores y target en un solo DataFrame para facilitar el EDA
df = pd.concat([X_raw, y_raw], axis=1)
 
# El dataset UCI entrega las columnas como X1..X23 e Y.
# Las renombramos a sus nombres originales del paper de Yeh & Lien (2009)
nuevos_nombres = {
    'X1':'LIMIT_BAL', 'X2':'SEX',       'X3':'EDUCATION', 'X4':'MARRIAGE',
    'X5':'AGE',       'X6':'PAY_0',     'X7':'PAY_2',     'X8':'PAY_3',
    'X9':'PAY_4',     'X10':'PAY_5',    'X11':'PAY_6',    'X12':'BILL_AMT1',
    'X13':'BILL_AMT2','X14':'BILL_AMT3','X15':'BILL_AMT4','X16':'BILL_AMT5',
    'X17':'BILL_AMT6','X18':'PAY_AMT1', 'X19':'PAY_AMT2', 'X20':'PAY_AMT3',
    'X21':'PAY_AMT4', 'X22':'PAY_AMT5', 'X23':'PAY_AMT6', 'Y':'default'
}
df.rename(columns=nuevos_nombres, inplace=True)
 
# Si el target viene como DataFrame de una columna, lo aplana a Serie
if isinstance(df['default'], pd.DataFrame):
    df['default'] = df['default'].iloc[:, 0]
 
print(f"  Dimensiones originales: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"  Tipos de datos únicos : {df.dtypes.unique()}")
print(f"  Memoria               : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
 
 
# ==============================================================================
# BLOQUE 2 — ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================
print("\n" + "=" * 65)
print("BLOQUE 2 — EDA")
print("=" * 65)

# --- CORRECCIÓN SUSTANTIVA: Eliminar duplicados ANTES del EDA ---
n_antes = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  ✓ Duplicados eliminados antes del EDA: {n_antes - len(df)} → {len(df):,} observaciones limpias.\n")


# ── 2.1 Dimensiones, tipos y valores nulos ─────────────────────────────────
print("\n── 2.1  Resumen general ──")
print(df.info())
print(f"\n  Total valores nulos : {df.isnull().sum().sum()}")
print(f"  Filas duplicadas    : {df.duplicated().sum()}")
 
# ── 2.2 Estadísticas descriptivas ─────────────────────────────────────────
print("\n── 2.2  Estadísticas descriptivas ──")
print(df.describe().T.round(2))
 
# ── 2.3 Desbalance de clases ────────────────────────────────────────────────
# Es CRÍTICO analizar el desbalance antes de modelar.
# Un clasificador trivial que siempre predice "0" obtiene ~78% de accuracy,
# lo que parece bueno pero es inútil. Por eso necesitamos AUC y F1.
conteo  = df['default'].value_counts()
propor  = df['default'].value_counts(normalize=True) * 100
ratio   = conteo[0] / conteo[1]
 
print("\n── 2.3  Desbalance de clases ──")
print(pd.DataFrame({'Conteo': conteo, 'Porcentaje': propor.round(2)}))
print(f"  Ratio mayoritaria/minoritaria: {ratio:.2f}:1")
# Un ratio ~3.5:1 se considera desbalance MODERADO.
# Por encima de 10:1 se considera severo y requiere técnicas más agresivas.
 
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(['No incumple (0)', 'Incumple (1)'], conteo.values,
            color=[COLOR_NEG, COLOR_POS], edgecolor='white')
axes[0].set_title('Distribución de la variable objetivo')
axes[0].set_ylabel('Frecuencia')
for i, (v, p) in enumerate(zip(conteo.values, propor.values)):
    axes[0].text(i, v + 150, f'{v:,}\n({p:.1f}%)', ha='center', fontsize=10)
 
# Tasa de incumplimiento por nivel educativo
tasa_edu = df.groupby('EDUCATION')['default'].mean() * 100
axes[1].bar(tasa_edu.index.astype(str), tasa_edu.values,
            color=COLOR_M1, edgecolor='white')
axes[1].set_title('Tasa de incumplimiento por EDUCATION')
axes[1].set_ylabel('% incumplimiento')
axes[1].set_xlabel('Nivel educativo')
plt.tight_layout()
plt.savefig('eda_01_clases_y_educacion.png', dpi=150)
plt.show()
print("  → eda_01_clases_y_educacion.png guardada")
 
# ── 2.4 Variables categóricas con categorías no documentadas ────────────────
print("\n── 2.4  Categorías no documentadas ──")
print("  EDUCATION (válidas 1-4):")
print(df['EDUCATION'].value_counts().sort_index())
# Los valores 0, 5, 6 no están en el diccionario de datos oficial de UCI.
# Representan ~468 observaciones que hay que reclasificar.
 
print("\n  MARRIAGE (válidas 1-3):")
print(df['MARRIAGE'].value_counts().sort_index())
# El valor 0 tampoco está documentado (~54 observaciones).
 
# ── 2.5 Historial de pagos por clase ────────────────────────────────────────
# PAY_0 es el mes más reciente. Valores negativos = pagado/sin crédito.
# Valores positivos = meses de retraso. Es el predictor con mayor poder
# discriminativo visual entre clases.
pay_vars = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(pay_vars):
    for cls, color, label in [(0, COLOR_NEG, 'No incumple'), (1, COLOR_POS, 'Incumple')]:
        subset = df[df['default'] == cls][col]
        bins   = range(int(df[col].min()) - 1, int(df[col].max()) + 2)
        axes[i].hist(subset, bins=bins, alpha=0.65, color=color,
                     label=label, edgecolor='white', density=True)
    axes[i].set_title(col)
    axes[i].set_xlabel('Estado de pago')
    axes[i].set_ylabel('Densidad')
    axes[i].legend(fontsize=8)
plt.suptitle('Historial de pagos por clase (densidad)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('eda_02_historial_pagos.png', dpi=150)
plt.show()
print("  → eda_02_historial_pagos.png guardada")
 
# ── 2.6 Distribución de variables numéricas ─────────────────────────────────
num_vars = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2',
            'BILL_AMT3', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(num_vars):
    for cls, color, label in [(0, COLOR_NEG, 'No incumple'), (1, COLOR_POS, 'Incumple')]:
        axes[i].hist(df[df['default'] == cls][col], bins=40,
                     alpha=0.6, color=color, label=label, edgecolor='white')
    axes[i].set_title(col)
    axes[i].set_xlabel('Valor')
    axes[i].set_ylabel('Frecuencia')
    if i == 0:
        axes[i].legend(fontsize=8)
plt.suptitle('Distribución de variables numéricas por clase', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('eda_03_distribuciones_numericas.png', dpi=150)
plt.show()
print("  → eda_03_distribuciones_numericas.png guardada")
 
# ── 2.7 Matriz de correlación ────────────────────────────────────────────────
# Este paso es fundamental: si hay alta correlación entre predictores
# (multicolinealidad), la regresión logística clásica produce estimaciones
# inestables. LASSO resuelve esto penalizando y eliminando variables redundantes.
corr = df.drop(columns='default').corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
            annot=False, linewidths=0.3,
            cbar_kws={'shrink': 0.7}, ax=ax)
ax.set_title('Matriz de correlación entre predictores\n'
             '(rojo intenso = alta correlación positiva → riesgo de multicolinealidad)')
plt.tight_layout()
plt.savefig('eda_04_correlacion.png', dpi=150)
plt.show()
print("  → eda_04_correlacion.png guardada")
 
# Identificar pares con r > 0.80 (umbral habitual para multicolinealidad)
corr_abs = corr.abs()
pares_alta_corr = [
    (corr_abs.columns[i], corr_abs.columns[j], round(corr_abs.iloc[i, j], 3))
    for i in range(len(corr_abs.columns))
    for j in range(i + 1, len(corr_abs.columns))
    if corr_abs.iloc[i, j] > 0.80
]
print(f"\n  Pares con |r| > 0.80 ({len(pares_alta_corr)} encontrados):")
for v1, v2, r in sorted(pares_alta_corr, key=lambda x: -x[2]):
    print(f"    {v1} ↔ {v2}  r = {r}")
# Si hay muchos pares, confirma que LASSO es la técnica adecuada.
 
 
# ==============================================================================
# BLOQUE 3 — LIMPIEZA Y PREPROCESAMIENTO
# ==============================================================================
print("\n" + "=" * 65)
print("BLOQUE 3 — LIMPIEZA Y PREPROCESAMIENTO")
print("=" * 65)
 

 
# ── 3.2 Reclasificar categorías no documentadas ──────────────────────────────
# EDUCATION: el diccionario UCI solo define 1=graduado, 2=universidad,
#             3=preparatoria, 4=otros. Los valores 0, 5, 6 no tienen
#             descripción → los agrupamos en la categoría 4 ("otros").
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
 
# MARRIAGE: solo están documentados 1=casado, 2=soltero, 3=otros.
#            El valor 0 no tiene descripción → lo movemos a 3 ("otros").
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
 
print(f"  EDUCATION después de limpieza: {df['EDUCATION'].value_counts().sort_index().to_dict()}")
print(f"  MARRIAGE después de limpieza : {df['MARRIAGE'].value_counts().sort_index().to_dict()}")
 
# ── 3.3 Codificación de variables categóricas ────────────────────────────────
# SEX, EDUCATION y MARRIAGE son categóricas nominales/ordinales.
# Se convierten a variables indicadoras (dummies).
# drop_first=True elimina una categoría por variable para evitar
# multicolinealidad perfecta (trampa de las variables dummy).
# Categoría de referencia: SEX=1 (hombre), EDUCATION=1 (graduado), MARRIAGE=1 (casado)
df_model = pd.get_dummies(
    df,
    columns=['SEX', 'EDUCATION', 'MARRIAGE'],
    drop_first=True,
    dtype=int
)
 
TARGET       = 'default'
feature_cols = [c for c in df_model.columns if c != TARGET]
 
X_all = df_model[feature_cols].astype(float)
y_all = df_model[TARGET].astype(int)
 
print(f"\n  Variables predictoras tras codificación: {len(feature_cols)}")
print(f"  Nombres: {feature_cols}")
 
 
# ==============================================================================
# BLOQUE 4 — ESTRATEGIA PARA EL DESBALANCE DE CLASES
# ==============================================================================
print("\n" + "=" * 65)
print("BLOQUE 4 — DESBALANCE DE CLASES Y ESTRATEGIA ADOPTADA")
print("=" * 65)

# (Deja los comentarios que tenías explicando por qué usas class_weight='balanced')
print(f"\n  Estrategia adoptada: class_weight='balanced' en sklearn para predicción.")
print(f"  Inferencia en statsmodels se mantendrá sin pesos para evitar distorsionar los SE.")
 
# ==============================================================================
# BLOQUE 5 — PARTICIÓN ESTRATIFICADA Y ESTANDARIZACIÓN
# ==============================================================================
print("\n" + "=" * 65)
print("BLOQUE 5 — PARTICIÓN Y ESTANDARIZACIÓN")
print("=" * 65)
 
# ── 5.1 Partición 70/30 estratificada ───────────────────────────────────────
# stratify=y_all garantiza que la proporción de incumplimiento (22%)
# se preserve EXACTAMENTE en train y en test.
# random_state=42 asegura reproducibilidad del experimento.
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.30,
    random_state=42,
    stratify=y_all
)
 
print(f"  Train: {X_train.shape[0]:,} obs  |  default rate: {y_train.mean()*100:.2f}%")
print(f"  Test : {X_test.shape[0]:,}  obs  |  default rate: {y_test.mean()*100:.2f}%")
 
# ── 5.2 Estandarización ─────────────────────────────────────────────────────
# Transforma cada variable a media=0 y desviación=1.
# Es OBLIGATORIA para LASSO: sin estandarizar, variables con magnitudes
# grandes (p.ej. BILL_AMT en miles) reciben penalización desproporcionada
# y el resultado de la selección de variables es inválido.
# También mejora la convergencia numérica de ambos modelos.
#
# REGLA CRÍTICA: fit SOLO sobre train, transform sobre train y test.
# Si usáramos estadísticos del test, estaríamos filtrando información
# del futuro hacia el entrenamiento (data leakage).
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # aprende media y DE de train
X_test_sc  = scaler.transform(X_test)        # aplica los mismos parámetros
 
print(f"\n  Estandarización aplicada (fit solo en train)")
print(f"  Media train post-scale (debe ser ≈0): {X_train_sc.mean():.6f}")
print(f"  DE   train post-scale (debe ser ≈1): {X_train_sc.std():.6f}")
 
# Guardamos objetos para reutilizarlos si se ejecuta el script por partes
joblib.dump(scaler,       'scaler_train.pkl')
joblib.dump(X_train_sc,   'X_train_sc.pkl')
joblib.dump(X_test_sc,    'X_test_sc.pkl')
joblib.dump(y_train,      'y_train.pkl')
joblib.dump(y_test,       'y_test.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')
 
 
# ==============================================================================
# BLOQUE 6 — MODELO M1: REGRESIÓN LOGÍSTICA CLÁSICA
# ==============================================================================
print("\n" + "=" * 65)
print("BLOQUE 6 — MODELO M1: REGRESIÓN LOGÍSTICA CLÁSICA")
print("=" * 65)

# --- 6.0 Modelo predictivo (sklearn, con class_weight para desbalance) -------
# Se entrena sobre los datos ESTANDARIZADOS (X_train_sc)
m1 = LogisticRegression(
    penalty=None,
    solver='lbfgs',
    max_iter=2000,
    class_weight='balanced',
    random_state=42
)
m1.fit(X_train_sc, y_train)
print("  ✓ Modelo M1 ajustado (sklearn, MLE con pesos de clase)")

# --- 6.1 Inferencia estadística (statsmodels) -------------------------------
# DECISIÓN METODOLÓGICA CORREGIDA:
# Ajustamos statsmodels SOBRE LOS DATOS ORIGINALES (X_train, no estandarizados) y SIN pesos.
# Esto hace que los Odds Ratios (OR) sean directamente interpretables en sus unidades originales.
X_train_inf = sm.add_constant(X_train.astype(float))  # SIN estandarizar
glm_sm    = sm.GLM(y_train, X_train_inf, family=sm.families.Binomial())
result_sm = glm_sm.fit()

col_sm   = ['Intercepto'] + list(X_train.columns)
conf_int = result_sm.conf_int()

coef_tabla = pd.DataFrame({
    'Variable'  : col_sm,
    'β'         : result_sm.params.values.round(4),
    'Err. est.' : result_sm.bse.values.round(4),
    'z'         : result_sm.tvalues.values.round(3),
    'p-valor'   : result_sm.pvalues.values.round(4),
    'OR'        : np.exp(result_sm.params.values).round(4),
    'IC inf 95%': np.exp(conf_int.iloc[:, 0].values).round(4),
    'IC sup 95%': np.exp(conf_int.iloc[:, 1].values).round(4),
}).sort_values('p-valor')

print("\n── Tabla de coeficientes M1 (ordenada por p-valor) ──")
print(coef_tabla.to_string(index=False))

# CORRECCIÓN DE ERROR 2: Conteo correcto excluyendo explícitamente el intercepto
n_sig = ((coef_tabla['p-valor'] < 0.05) & 
         (coef_tabla['Variable'] != 'Intercepto')).sum()
print(f"\n  Variables significativas (p < 0.05): {n_sig} de {len(feature_cols)}")

# ── 6.2 Gráfico de Odds Ratios significativos ───────────────────────────────
sig = coef_tabla[(coef_tabla['p-valor'] < 0.05) & (coef_tabla['Variable'] != 'Intercepto')]
sig = sig.sort_values('OR', ascending=True)

fig, ax = plt.subplots(figsize=(9, max(4, len(sig) * 0.38)))
colors_or = [COLOR_POS if v > 1 else COLOR_M1 for v in sig['OR']]
ax.barh(sig['Variable'], sig['OR'],
        xerr=[sig['OR'] - sig['IC inf 95%'], sig['IC sup 95%'] - sig['OR']],
        color=colors_or, edgecolor='white', capsize=3, height=0.65)
ax.axvline(1, color='black', lw=1.2, linestyle='--', label='OR = 1 (sin efecto)')
ax.set_xlabel('Odds Ratio con IC 95%')
ax.set_title('Odds Ratios — M1 Regresión Logística Clásica\n'
             'Rojo: factor de riesgo (OR > 1) | Azul: factor protector (OR < 1)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('m1_odds_ratios.png', dpi=150)
plt.show()
print("\n  → m1_odds_ratios.png guardada")

# ── 6.3 Evaluación en test externo ──────────────────────────────────────────
# Evaluamos el rendimiento predictivo usando el modelo de sklearn (m1) en los datos estandarizados
y_prob_m1 = m1.predict_proba(X_test_sc)[:, 1]
y_pred_m1 = m1.predict(X_test_sc)

auc_m1   = roc_auc_score(y_test, y_prob_m1)
f1_m1    = f1_score(y_test, y_pred_m1)
prec_m1  = precision_score(y_test, y_pred_m1)
rec_m1   = recall_score(y_test, y_pred_m1)
acc_m1   = accuracy_score(y_test, y_pred_m1)

print("\n── Métricas M1 en test externo ──")
print(f"  AUC-ROC   : {auc_m1:.4f}  ← métrica principal")
print(f"  F1-score  : {f1_m1:.4f}")
print(f"  Precision : {prec_m1:.4f}")
print(f"  Recall    : {rec_m1:.4f}")
print(f"  Accuracy  : {acc_m1:.4f}")

print("\n── Reporte de clasificación (M1) ──")
print(classification_report(y_test, y_pred_m1,
                            target_names=['No incumple (0)', 'Incumple (1)']))

# Matriz de confusión M1
cm1  = confusion_matrix(y_test, y_pred_m1)
disp = ConfusionMatrixDisplay(cm1, display_labels=['No incumple', 'Incumple'])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Matriz de confusión — Modelo M1 (Logística Clásica)')
plt.tight_layout()
plt.savefig('m1_confusion_matrix.png', dpi=150)
plt.show()
print("  → m1_confusion_matrix.png guardada")

# Guardamos las variables necesarias para cuando decidas correr los bloques comparativos
joblib.dump(y_prob_m1, 'y_prob_m1.pkl')
joblib.dump(auc_m1, 'auc_m1.pkl')
 


 # ==============================================================================
# BLOQUE 7 — MODELO M2: REGRESIÓN LOGÍSTICA CON PENALIZACIÓN LASSO
# ==============================================================================
print("\n" + "=" * 65)
print("BLOQUE 7 — MODELO M2: REGRESIÓN LOGÍSTICA CON LASSO (L1)")
print("=" * 65)

# JUSTIFICACIÓN:
# LASSO añade a la verosimilitud una penalización L1 (Σ|βj|) que:
#   1) Realiza selección automática de variables (β exactamente = 0).
#   2) Maneja la multicolinealidad detectada en el EDA (BILL_AMT con r > 0.90).
#   3) Reduce varianza y mejora la generalización.
# El hiperparámetro C (=1/λ) se elige por CV estratificada 10-fold maximizando AUC.

# ── 7.1 Ajuste de M2 con selección de C por CV estratificada ─────────────
cv_strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

m2 = LogisticRegressionCV(
    Cs=50,
    cv=cv_strat,
    penalty='l1',
    solver='saga',
    scoring='roc_auc',
    class_weight='balanced',
    max_iter=5000,
    n_jobs=-1,
    refit=True,
    random_state=42
)
m2.fit(X_train_sc, y_train)

C_opt      = m2.C_[0]
lambda_opt = 1 / C_opt
print(f"  ✓ Modelo M2 ajustado (LASSO, saga, 10-fold CV estratificado)")
print(f"  C óptimo (1/λ) : {C_opt:.6f}")
print(f"  λ equivalente  : {lambda_opt:.6f}")

# ── 7.2 Coeficientes y variables seleccionadas ───────────────────────────
coef_m2 = pd.DataFrame({
    'Variable'    : feature_cols,
    'β (estand.)' : m2.coef_.ravel().round(4),
    'OR'          : np.exp(m2.coef_.ravel()).round(4),
    'Seleccionada': np.where(m2.coef_.ravel() != 0, 'Sí', 'No')
})
coef_m2['|β|'] = coef_m2['β (estand.)'].abs()
coef_m2 = coef_m2.sort_values('|β|', ascending=False).drop(columns='|β|')

n_sel  = (m2.coef_.ravel() != 0).sum()
n_drop = (m2.coef_.ravel() == 0).sum()
print(f"\n  Variables seleccionadas (β ≠ 0): {n_sel} de {len(feature_cols)}")
print(f"  Variables eliminadas    (β = 0): {n_drop}")
print("\n── Coeficientes M2 (ordenados por |β|) ──")
print(coef_m2.to_string(index=False))

# ── 7.3 Trayectoria de coeficientes vs log(λ) ────────────────────────────
Cs_grid = np.logspace(-4, 4, 50)
coefs_path = []
for C in Cs_grid:
    mod = LogisticRegression(
        penalty='l1', solver='saga', C=C,
        class_weight='balanced', max_iter=5000, random_state=42
    )
    mod.fit(X_train_sc, y_train)
    coefs_path.append(mod.coef_.ravel())
coefs_path = np.array(coefs_path)

fig, ax = plt.subplots(figsize=(11, 6))
for j, name in enumerate(feature_cols):
    ax.plot(np.log10(1 / Cs_grid), coefs_path[:, j], lw=1.2, alpha=0.8, label=name)
ax.axvline(np.log10(lambda_opt), color='black', ls='--', lw=1.3,
           label=f'log10(λ óptimo) = {np.log10(lambda_opt):.2f}')
ax.axhline(0, color='gray', lw=0.6)
ax.set_xlabel('log10(λ)')
ax.set_ylabel('Coeficiente estandarizado β')
ax.set_title('Trayectoria de coeficientes LASSO\n'
             'A mayor λ, más coeficientes se contraen exactamente a 0')
ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.savefig('m2_lasso_path.png', dpi=150)
plt.show()
print("\n  → m2_lasso_path.png guardada")

# ── 7.4 Curva de AUC promedio (CV) vs C ──────────────────────────────────
auc_cv_mean = m2.scores_[1].mean(axis=0)
auc_cv_std  = m2.scores_[1].std(axis=0)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(np.log10(m2.Cs_), auc_cv_mean, color=COLOR_M2, lw=2, label='AUC medio (CV)')
ax.fill_between(np.log10(m2.Cs_),
                auc_cv_mean - auc_cv_std,
                auc_cv_mean + auc_cv_std,
                color=COLOR_M2, alpha=0.20, label='± 1 DE')
ax.axvline(np.log10(C_opt), color='black', ls='--', lw=1.3,
           label=f'C óptimo = {C_opt:.4f}')
ax.set_xlabel('log10(C) — C = 1/λ')
ax.set_ylabel('AUC-ROC (10-fold CV)')
ax.set_title('Selección del hiperparámetro C por validación cruzada')
ax.legend()
plt.tight_layout()
plt.savefig('m2_cv_auc.png', dpi=150)
plt.show()
print("  → m2_cv_auc.png guardada")

# ── 7.5 Gráfico de coeficientes seleccionados ────────────────────────────
sel = coef_m2[coef_m2['Seleccionada'] == 'Sí'].sort_values('β (estand.)')
fig, ax = plt.subplots(figsize=(9, max(4, len(sel) * 0.38)))
colors_b = [COLOR_POS if v > 0 else COLOR_M1 for v in sel['β (estand.)']]
ax.barh(sel['Variable'], sel['β (estand.)'], color=colors_b, edgecolor='white', height=0.65)
ax.axvline(0, color='black', lw=1.2, linestyle='--')
ax.set_xlabel('Coeficiente estandarizado β')
ax.set_title(f'M2 LASSO — {n_sel} variables seleccionadas\n'
             'Rojo: factor de riesgo (β > 0) | Azul: factor protector (β < 0)')
plt.tight_layout()
plt.savefig('m2_coeficientes.png', dpi=150)
plt.show()
print("  → m2_coeficientes.png guardada")

# ── 7.6 Evaluación en test externo ───────────────────────────────────────
y_prob_m2 = m2.predict_proba(X_test_sc)[:, 1]
y_pred_m2 = m2.predict(X_test_sc)

auc_m2  = roc_auc_score(y_test, y_prob_m2)
f1_m2   = f1_score(y_test, y_pred_m2)
prec_m2 = precision_score(y_test, y_pred_m2)
rec_m2  = recall_score(y_test, y_pred_m2)
acc_m2  = accuracy_score(y_test, y_pred_m2)

print("\n── Métricas M2 en test externo ──")
print(f"  AUC-ROC   : {auc_m2:.4f}  ← métrica principal")
print(f"  F1-score  : {f1_m2:.4f}")
print(f"  Precision : {prec_m2:.4f}")
print(f"  Recall    : {rec_m2:.4f}")
print(f"  Accuracy  : {acc_m2:.4f}")

print("\n── Reporte de clasificación (M2) ──")
print(classification_report(y_test, y_pred_m2,
                            target_names=['No incumple (0)', 'Incumple (1)']))

# Matriz de confusión M2
cm2  = confusion_matrix(y_test, y_pred_m2)
disp = ConfusionMatrixDisplay(cm2, display_labels=['No incumple', 'Incumple'])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap='Reds', colorbar=False)
ax.set_title('Matriz de confusión — Modelo M2 (LASSO)')
plt.tight_layout()
plt.savefig('m2_confusion_matrix.png', dpi=150)
plt.show()
print("  → m2_confusion_matrix.png guardada")

# ── 7.7 Comparación preliminar M1 vs M2 ──────────────────────────────────
resumen = pd.DataFrame({
    'Métrica'      : ['AUC-ROC', 'F1', 'Precision', 'Recall', 'Accuracy', 'N° variables'],
    'M1 (Clásica)' : [auc_m1, f1_m1, prec_m1, rec_m1, acc_m1, len(feature_cols)],
    'M2 (LASSO)'   : [auc_m2, f1_m2, prec_m2, rec_m2, acc_m2, n_sel]
})
print("\n── Comparación preliminar M1 vs M2 ──")
print(resumen.to_string(index=False))

# Serialización para el bloque comparativo (DeLong, ROC superpuestas)
joblib.dump(m2,        'm2_lasso.pkl')
joblib.dump(y_prob_m2, 'y_prob_m2.pkl')
joblib.dump(auc_m2,    'auc_m2.pkl')
joblib.dump(coef_m2,   'coef_m2.pkl')
print("\n  ✓ Objetos M2 guardados")

# ==============================================================================
# BLOQUE 8B — EXPLORACIÓN MANUAL DE LAMBDA PARA ANÁLISIS FINANCIERO
# ==============================================================================
# Requiere Bloques 0-7 ya ejecutados (usa X_train_sc, y_train, X_test_sc,
# y_test, feature_cols, auc_m1, coef_m2, COLOR_M2 ya definidos).

print("\n" + "=" * 65)
print("BLOQUE 8B — LAMBDA MANUAL PARA JUSTIFICACIÓN FINANCIERA")
print("=" * 65)

# ── 8B.1 Define aquí los valores de C que quieras probar a mano ─────────────
# Recuerda: C = 1/λ. Mientras MÁS CHICO es C, MÁS GRANDE es λ,
# y MÁS variables se eliminan (más penalización).
# El óptimo de tu CV fue C=10000 (λ=0.0001, no elimina nada).
# Aquí exploramos valores más chicos, en orden decreciente de C.
Cs_manual = [10000, 100, 10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

resultados_manual = []
detalle_variables  = {}

for C in Cs_manual:
    mod = LogisticRegression(
        penalty='l1', solver='saga', C=C,
        class_weight='balanced', max_iter=5000, random_state=42
    )
    mod.fit(X_train_sc, y_train)

    coefs = mod.coef_.ravel()
    seleccionadas = [feature_cols[i] for i in range(len(feature_cols)) if coefs[i] != 0]
    eliminadas    = [feature_cols[i] for i in range(len(feature_cols)) if coefs[i] == 0]

    prob = mod.predict_proba(X_test_sc)[:, 1]
    auc_c = roc_auc_score(y_test, prob)
    pred_c = mod.predict(X_test_sc)
    f1_c   = f1_score(y_test, pred_c)

    resultados_manual.append({
        'C': C,
        'lambda': round(1/C, 5),
        'n_variables_activas': len(seleccionadas),
        'n_eliminadas': len(eliminadas),
        'AUC_test': round(auc_c, 4),
        'delta_AUC_vs_M1': round(auc_c - auc_m1, 4),
        'F1_test': round(f1_c, 4)
    })
    detalle_variables[C] = {'seleccionadas': seleccionadas, 'eliminadas': eliminadas}

tabla_manual = pd.DataFrame(resultados_manual)
print("\n── Tabla comparativa: distintos lambda manuales ──")
print(tabla_manual.to_string(index=False))
tabla_manual.to_csv('tabla_lambda_manual.csv', index=False)
print("\n  → tabla_lambda_manual.csv guardada")

# ── 8B.2 Detalle de qué variables elimina cada C ─────────────────────────────
print("\n── Detalle: variables eliminadas por cada valor de C ──")
for C in Cs_manual:
    elim = detalle_variables[C]['eliminadas']
    print(f"\n  C={C}  (λ={1/C:.5f})  → {len(elim)} variables eliminadas:")
    print(f"    {elim if elim else '(ninguna)'}")

# ── 8B.3 Elige TU lambda final para el análisis financiero ─────────────────
# Una vez que revises la tabla de arriba, escoge el C que te parezca el
# mejor punto de equilibrio (ej. el que elimina las variables de BILL_AMT
# redundantes sin sacrificar más de X puntos de AUC) y ajusta el modelo
# final con ese valor.

C_FINANCIERO = 0.01   # <-- CAMBIA ESTE VALOR según lo que decidas de la tabla

m2_financiero = LogisticRegression(
    penalty='l1', solver='saga', C=C_FINANCIERO,
    class_weight='balanced', max_iter=5000, random_state=42
)
m2_financiero.fit(X_train_sc, y_train)

coef_fin = pd.DataFrame({
    'Variable': feature_cols,
    'β (estand.)': m2_financiero.coef_.ravel().round(4),
    'OR': np.exp(m2_financiero.coef_.ravel()).round(4)
})
coef_fin['Seleccionada'] = np.where(coef_fin['β (estand.)'] != 0, 'Sí', 'No')
coef_fin = coef_fin.reindex(coef_fin['β (estand.)'].abs().sort_values(ascending=False).index)

n_sel_fin = (m2_financiero.coef_.ravel() != 0).sum()
y_prob_fin = m2_financiero.predict_proba(X_test_sc)[:, 1]
y_pred_fin = m2_financiero.predict(X_test_sc)
auc_fin  = roc_auc_score(y_test, y_prob_fin)
f1_fin   = f1_score(y_test, y_pred_fin)
prec_fin = precision_score(y_test, y_pred_fin)
rec_fin  = recall_score(y_test, y_pred_fin)

# ── 8B.5 Matriz de confusión para el modelo financiero final ──
# Esta es la matriz solicitada con la selección de variables ajustada
cm_fin = confusion_matrix(y_test, y_pred_fin)
disp_fin = ConfusionMatrixDisplay(cm_fin, display_labels=['No incumple', 'Incumple'])

fig, ax = plt.subplots(figsize=(5, 4))
disp_fin.plot(ax=ax, cmap='Greens', colorbar=False) # Usamos verde para diferenciar
ax.set_title(f'Matriz de confusión — M2 (LASSO Financiero)\n(C={C_FINANCIERO}, vars={n_sel_fin})')
plt.tight_layout()
plt.savefig('m4_confusion_matrix_financiero.png', dpi=150)
plt.show()
print("  → m4_confusion_matrix_financiero.png guardada")

print(f"\n{'='*65}")
print(f"MODELO FINAL — C={C_FINANCIERO}  (λ={1/C_FINANCIERO:.5f})")
print(f"{'='*65}")
print(f"  Variables retenidas   : {n_sel_fin} de {len(feature_cols)}")
print(f"  AUC-ROC (test)        : {auc_fin:.4f}   (M1 = {auc_m1:.4f}, Δ = {auc_fin-auc_m1:+.4f})")
print(f"  F1-score (test)       : {f1_fin:.4f}")
print(f"  Precision (test)      : {prec_fin:.4f}")
print(f"  Recall (test)         : {rec_fin:.4f}")
print("\n── Coeficientes del modelo financiero (ordenados por |β|) ──")
print(coef_fin.to_string(index=False))

# ── 8B.4 Gráfico comparativo de los 3 modelos: M1, M2 (óptimo CV), M2 financiero ──
fig, ax = plt.subplots(figsize=(9, 5))
modelos_comp = ['M1\n(Clásico)', 'M2\n(LASSO óptimo CV)', f'M2\n(LASSO financiero,\nC={C_FINANCIERO})']
aucs_comp    = [auc_m1, auc_m2, auc_fin]
nvars_comp   = [len(feature_cols), n_sel, n_sel_fin]

ax2 = ax.twinx()
bars = ax.bar(modelos_comp, aucs_comp, color=[COLOR_M1, COLOR_M2, '#F2A93B'],
              alpha=0.75, edgecolor='white', width=0.5)
ax.set_ylabel('AUC-ROC en test')
ax.set_ylim(0.5, 0.8)
for i, v in enumerate(aucs_comp):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

ax2.plot(modelos_comp, nvars_comp, 'o--', color='black', markersize=8, label='N° variables')
for i, v in enumerate(nvars_comp):
    ax2.text(i, v + 0.6, f'{v} vars', ha='center', color='black', fontsize=9)
ax2.set_ylabel('Número de variables activas')
ax2.set_ylim(0, len(feature_cols) + 3)

ax.set_title('Comparación final: desempeño vs. parsimonia\n'
             'Base para la justificación financiera de la elección de λ')
plt.tight_layout()
plt.savefig('m4_comparacion_final_financiero.png', dpi=150)
plt.show()
print("\n  → m4_comparacion_final_financiero.png guardada")

joblib.dump(m2_financiero, 'm2_lasso_financiero.pkl')
joblib.dump(coef_fin, 'coef_financiero.pkl')
print("\n  ✓ Objetos del Bloque 8B guardados")


# ==============================================================================
# BLOQUE 9 — PRUEBA DE DELONG (M1 vs M2)
# ==============================================================================
# JUSTIFICACIÓN:
# La prueba de DeLong, compara dos
# curvas ROC correlacionadas (evaluadas sobre EL MISMO conjunto de prueba,
# como es nuestro caso) y determina si la diferencia entre sus AUC es
# estadísticamente significativa. No está implementada en scipy/sklearn,
# así que se programa siguiendo el algoritmo "fast DeLong" ,
# que es la formalización computacional eficiente del método original.
#
# Esto corresponde exactamente al "Filtro 1" de tu Regla de Decisión (III-E).
 
print("\n" + "=" * 65)
print("BLOQUE 9 — PRUEBA DE DELONG")
print("=" * 65)
 
def compute_midrank(x):
    """Calcula los mid-ranks de un vector, necesarios para el cómputo
    de varianzas de DeLong (maneja empates correctamente)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    predictions_sorted_transposed: array (k modelos, n observaciones),
        con las observaciones ordenadas de forma que las positivas
        (default=1) vienen primero.
    label_1_count: número de observaciones positivas (default=1).
    Retorna: AUCs de cada modelo y su matriz de covarianza (DeLong).
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_A, prob_B, alpha=0.05):
    """
    Compara los AUC de dos modelos (prob_A, prob_B) evaluados sobre el
    MISMO y_true. Retorna z, p-valor, AUCs, delta_AUC, error estándar
    y el intervalo de confianza (1-alpha) para AUC_B - AUC_A.
    """
    y_true = np.asarray(y_true)
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    predictions_sorted_transposed = np.vstack([prob_A, prob_B])[:, order]

    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    l_vec = np.array([[1, -1]])
    var = np.dot(np.dot(l_vec, delongcov), l_vec.T)[0, 0]
    se = np.sqrt(var)

    z = (aucs[0] - aucs[1]) / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))

    delta_auc = aucs[1] - aucs[0]
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ic_inf = delta_auc - z_crit * se
    ic_sup = delta_auc + z_crit * se

    return z, p, aucs, delta_auc, se, (ic_inf, ic_sup)


# ── 9.1 Comparación central: M1 vs M2 (Filtro 1 de la Regla de Decisión) ────
z_12, p_12, aucs_12, delta_12, se_12, ic_12 = delong_roc_test(
    y_test.values, y_prob_m1, y_prob_m2
)

print("\n── DeLong: M1 (Clásica) vs M2 (LASSO óptimo por CV) ──")
print(f"  AUC M1        : {aucs_12[0]:.4f}")
print(f"  AUC M2        : {aucs_12[1]:.4f}")
print(f"  ΔAUC          : {delta_12:+.4f}")
print(f"  Error estándar: {se_12:.4f}")
print(f"  IC 95% ΔAUC   : [{ic_12[0]:+.4f}, {ic_12[1]:+.4f}]")
print(f"  Estadístico Z : {z_12:.4f}")
print(f"  p-valor       : {p_12:.4f}")
print(f"  ¿Significativo (α=0.05)? {'Sí' if p_12 < 0.05 else 'No'}")
print(f"  (detalle) ΔAUC = {delta_12:.6f}   IC 95% = [{ic_12[0]:.6f}, {ic_12[1]:.6f}]")

# ── 9.2 Comparación adicional: M1 vs M2 financiero (C=0.01, 16 vars) ────────
z_1fin, p_1fin, aucs_1fin, delta_1fin, se_1fin, ic_1fin = delong_roc_test(
    y_test.values, y_prob_m1, y_prob_fin
)

print("\n── DeLong: M1 (Clásica) vs M2 financiero (C=0.01, 16 vars) ──")
print(f"  AUC M1            : {aucs_1fin[0]:.4f}")
print(f"  AUC M2 financiero : {aucs_1fin[1]:.4f}")
print(f"  ΔAUC              : {delta_1fin:+.4f}")
print(f"  Error estándar    : {se_1fin:.4f}")
print(f"  IC 95% ΔAUC       : [{ic_1fin[0]:+.4f}, {ic_1fin[1]:+.4f}]")
print(f"  Estadístico Z     : {z_1fin:.4f}")
print(f"  p-valor           : {p_1fin:.4f}")
print(f"  ¿Significativo (α=0.05)? {'Sí' if p_1fin < 0.05 else 'No'}")

# Guardamos resultados para redactar el Resultado Central (IV-D) con precisión
delong_resultados = pd.DataFrame({
    'Comparación': ['M1 vs M2 (LASSO óptimo)', 'M1 vs M2 financiero (C=0.01)'],
    'AUC_A': [aucs_12[0], aucs_1fin[0]],
    'AUC_B': [aucs_12[1], aucs_1fin[1]],
    'Delta_AUC': [delta_12, delta_1fin],
    'SE': [se_12, se_1fin],
    'IC_inf_95': [ic_12[0], ic_1fin[0]],
    'IC_sup_95': [ic_12[1], ic_1fin[1]],
    'Z': [z_12, z_1fin],
    'p_valor': [p_12, p_1fin]
})
delong_resultados.to_csv('delong_resultados.csv', index=False)
print("\n  → delong_resultados.csv guardada (usar estos números en IV-D)")
 
 
# ==============================================================================
# BLOQUE 10 — GRÁFICOS FALTANTES
# ==============================================================================
print("\n" + "=" * 65)
print("BLOQUE 10 — GRÁFICOS COMPARATIVOS FALTANTES")
print("=" * 65)
 
# ── 10.1 Curvas ROC superpuestas: M1 vs M2 vs M2 financiero ────────────────
fpr1, tpr1, _ = roc_curve(y_test, y_prob_m1)
fpr2, tpr2, _ = roc_curve(y_test, y_prob_m2)
fpr3, tpr3, _ = roc_curve(y_test, y_prob_fin)
 
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr1, tpr1, color=COLOR_M1, lw=2,
        label=f'M1 Clásica (AUC = {auc_m1:.4f})')
ax.plot(fpr2, tpr2, color=COLOR_M2, lw=2, linestyle='--',
        label=f'M2 LASSO óptimo CV (AUC = {auc_m2:.4f})')
ax.plot(fpr3, tpr3, color='#F2A93B', lw=2, linestyle=':',
        label=f'M2 LASSO financiero, C={C_FINANCIERO} (AUC = {auc_fin:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='-.',
        label='Clasificador aleatorio (AUC = 0.50)')
ax.set_xlabel('Tasa de falsos positivos (1 - Especificidad)')
ax.set_ylabel('Tasa de verdaderos positivos (Sensibilidad)')
ax.set_title('Curvas ROC — Comparación de modelos\n'
             f'DeLong M1 vs M2: p-valor = {p_12:.4f}')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('comparacion_curvas_roc.png', dpi=150)
plt.show()
print("  → comparacion_curvas_roc.png guardada")
 
# ── 10.2 AUC vs λ (a partir de la tabla del Bloque 8B) ──────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax2 = ax.twinx()
 
ax.plot(tabla_manual['lambda'], tabla_manual['AUC_test'],
        'o-', color=COLOR_M2, lw=2, markersize=6, label='AUC-ROC (test)')
ax.axhline(auc_m1, color=COLOR_M1, lw=1.3, linestyle='--',
           label=f'AUC M1 = {auc_m1:.4f} (referencia)')
ax.set_xscale('log')
ax.set_xlabel('λ (escala logarítmica)')
ax.set_ylabel('AUC-ROC', color=COLOR_M2)
ax.tick_params(axis='y', labelcolor=COLOR_M2)
 
ax2.plot(tabla_manual['lambda'], tabla_manual['n_variables_activas'],
         's--', color='black', lw=1.3, markersize=6, alpha=0.7,
         label='N° variables activas')
ax2.set_ylabel('N° variables activas', color='black')
 
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=8)
 
ax.set_title('Sensibilidad del AUC y la parsimonia ante el aumento de λ\n'
             'Punto de quiebre: a mayor penalización, el AUC decae y se pierden variables')
plt.tight_layout()
plt.savefig('sensibilidad_auc_vs_lambda.png', dpi=150)
plt.show()
print("  → sensibilidad_auc_vs_lambda.png guardada")
 
