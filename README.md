# 📡 TelecomX LATAM — Análisis Predictivo de Cancelación de Clientes

Análisis end-to-end sobre el dataset de TelecomX LATAM para identificar los factores que causan la cancelación de clientes y construir un modelo predictivo con Árbol de Decisión.

---

## 📋 Tabla de Contenidos

1. [Requisitos e Instalación](#requisitos-e-instalación)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Pipeline del Notebook](#pipeline-del-notebook)
4. [Problemas Encontrados y Soluciones](#problemas-encontrados-y-soluciones)
5. [Resultados del Análisis Exploratorio](#resultados-del-análisis-exploratorio)
6. [Resultados del Modelo ML](#resultados-del-modelo-ml)
7. [Variables Más Relevantes](#variables-más-relevantes)
8. [Conclusiones y Estrategias de Retención](#conclusiones-y-estrategias-de-retención)

---

## ⚙️ Requisitos e Instalación

### Python
Versión recomendada: **Python 3.10+**

### Librerías necesarias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Librería | Versión mínima | Uso |
|---|---|---|
| `pandas` | 1.5+ | Manipulación de datos |
| `numpy` | 1.23+ | Operaciones numéricas |
| `matplotlib` | 3.6+ | Visualizaciones |
| `seaborn` | 0.12+ | Gráficos estadísticos |
| `scikit-learn` | 1.2+ | Modelos ML y métricas |

### Ejecución

```bash
jupyter notebook TelecomX_LATAM_fixed.ipynb
```

### Fuente de datos

El dataset se carga directamente desde GitHub, no requiere descarga manual:

```python
df = pd.read_json("https://raw.githubusercontent.com/alura-cursos/challenge2-data-science-LATAM/refs/heads/main/TelecomX_Data.json")
```

---

## 🗂️ Estructura del Proyecto

```
TelecomX_LATAM/
│
├── TelecomX_LATAM2.ipynb   # Notebook principal corregido
└── README.md                    # Este archivo
```

---

## 🔄 Pipeline del Notebook

El notebook está dividido en 4 grandes etapas:

### 📌 1. Extracción
- Carga del JSON desde URL con `pd.read_json()`
- Desanidado de columnas estructuradas (`customer`, `phone`, `internet`, `account`) usando `pd.json_normalize()`
- Concatenación en un único DataFrame normalizado de **7.267 filas × 21 columnas**

### 🔧 2. Transformación
- Renombrado de columnas al español
- Estandarización de valores categóricos (`Yes/No` → `1/0`, traducciones de contratos y métodos de pago)
- Creación de la columna `Cuentas_diarias` = `Cargo_Mensual / 30`
- Creación de `antiguedad_bin` con `pd.cut()` en rangos `[0,18)`, `[18,36)`, `[36,54)`, `[54,72)`
- Limpieza de registros con `Churn` vacío y `Charges.Total` no numérico
- Reset de índice para garantizar continuidad

### 📊 3. Análisis Exploratorio (EDA)
- Estadísticas descriptivas con `df.describe()`
- Distribución de cancelación (churn 26.58%)
- Análisis por variables categóricas (contrato, método de pago, servicio de internet, etc.)
- Histograma de Cargo Total por grupo de cancelación
- Tasa de cancelación por rango de antigüedad
- Matriz de correlación entre variables numéricas

### 🤖 4. Machine Learning
- Preparación del DataFrame `df_ml`
- Conversión de intervalos y categorías a string
- Encoding con `OneHotEncoder` + `make_column_transformer`
- Split 80/20 estratificado
- Baseline con `DummyClassifier`
- Modelo principal: `DecisionTreeClassifier(max_depth=5)`
- Validación cruzada con `StratifiedKFold(n_splits=5)`
- Evaluación con métricas completas y matrices de confusión

---

## 🐛 Troubleshooting — Errores Frecuentes al Ejecutar el Código

Si al copiar y ejecutar el notebook te aparece alguno de estos errores, aquí está la solución directa.

---

### ❌ `ValueError: could not convert string to float: '[0, 18)'`

**¿Dónde aparece?** Al ejecutar `arbol.fit(X_train_final, Y_train)`

**¿Por qué pasa?** La columna `antiguedad_bin` se crea con `pd.cut()` y genera rangos como `[0, 18)`. Si no se la incluye en el `OneHotEncoder`, el árbol recibe texto que no puede procesar.

**Solución:** Asegúrate de que `antiguedad_bin` esté en la lista de columnas categóricas:

```python
# ✅ Así debe quedar
categoricas = ["Servicio_Internet", "Tipo_Contrato", "Método_Pago", "antiguedad_bin"]
```

---

### ❌ `ValueError: The palette dictionary is missing keys: {'1', '0'}`

**¿Dónde aparece?** En los `sns.boxplot()` de visualización

**¿Por qué pasa?** La columna `Abandonó_Servicio` puede quedar con valores string `'0'`/`'1'` en vez de enteros, y seaborn no encuentra las claves del diccionario de colores.

**Solución:** Usa una lista en vez de un diccionario para la paleta:

```python
# ❌ No usar
colores = {0: "#3498db", 1: "#e74c3c"}

# ✅ Usar esto
colores = ["#3498db", "#e74c3c"]
```

---

### ⚠️ `FutureWarning: Passing palette without assigning hue is deprecated`

**¿Dónde aparece?** En cualquier `sns.boxplot()` — no rompe el código pero aparecerá en versiones de seaborn 0.13+

**Solución:** Agregar `hue=` con la misma variable del eje X y `legend=False`:

```python
sns.boxplot(
    data=df_ml, x="Abandonó_Servicio", y="Cargo_Total",
    hue="Abandonó_Servicio", palette=colores,  # ← agregar hue
    legend=False, ax=axes[0, 0]                 # ← y legend=False
)
```

---

### ⚠️ `UserWarning: set_ticklabels() should only be used with a fixed number of ticks`

**¿Dónde aparece?** Después de los boxplots al llamar `set_xticklabels()`

**Solución:** Agregar `set_xticks()` en la línea anterior:

```python
axes[0, 0].set_xticks([0, 1])                          # ← esta línea primero
axes[0, 0].set_xticklabels(["No canceló", "Canceló"])  # ← luego esta
```

---

### ❌ `NameError: name 'X_train_final' is not defined`

**¿Dónde aparece?** Al ejecutar el `DummyClassifier` o el árbol

**¿Por qué pasa?** Las celdas no se ejecutaron en orden. `X_train_final` se crea en la celda de transformación (después del split).

**Solución:** Ejecutar las celdas en orden desde el inicio. En Jupyter: `Kernel → Restart & Run All`

---

### ❌ `KeyError: 'antiguedad_bin'` o `KeyError: 'Abandonó_Servicio'`

**¿Dónde aparece?** En la separación de X e Y o en el encoder

**¿Por qué pasa?** La columna `antiguedad_bin` se crea dentro de la función `abandonaron()` (celda 64). Si esa celda no se ejecutó antes del bloque ML, la columna no existe en `df_normalizado`.

**Solución:** Asegúrate de ejecutar la celda que contiene:

```python
df_normalizado["antiguedad_bin"] = pd.cut(df_normalizado['Antigüedad'], bins=[0, 18, 36, 54, 72], right=False)
```
antes de iniciar el pipeline de ML.

---

### ❌ `ConnectionError` o `JSONDecodeError` al cargar los datos

**¿Dónde aparece?** En la primera celda de carga del dataset

**¿Por qué pasa?** No hay conexión a internet o el repositorio de GitHub está caído.

**Solución:** Descarga el archivo manualmente desde [este enlace](https://raw.githubusercontent.com/alura-cursos/challenge2-data-science-LATAM/refs/heads/main/TelecomX_Data.json), guárdalo como `TelecomX_Data.json` en la misma carpeta del notebook y cámbialo así:

```python
# ✅ Carga local como alternativa
df = pd.read_json("TelecomX_Data.json")
```

---

## 📊 Resultados del Análisis Exploratorio

### Tasa de cancelación general
- **26.58%** de clientes cancelaron el servicio (~1 de cada 4)

### Perfiles de mayor riesgo

| Factor | Tasa de Cancelación |
|---|---|
| Cheque electrónico | 45.3% |
| Contrato mensual | 42.7% |
| Fibra óptica | 41.9% |
| Cliente senior | 41.7% |
| Primeros 18 meses | ~48% |

### Hallazgos clave
- Clientes que cancelan tienen **gasto total promedio de $1.531** vs $3.355 de los que se quedan
- Clientes que cancelan pagan **más mensualmente** ($74 vs $61), indicando que se van antes de amortizar el servicio
- La cancelación cae drásticamente después del mes 36 (baja a < 15%)
- Ausencia de Soporte Técnico y Seguridad Online correlaciona con mayor cancelación

---

## 🤖 Resultados del Modelo ML

### Comparación de modelos

| Modelo | Exactitud | Precisión | Recall | F1-Score |
|---|---|---|---|---|
| Baseline (Dummy) | 73.4% | 0.0% | 0.0% | 0.0% |
| Árbol — Test | 76.7% | 56.5% | 53.2% | 54.8% |
| Árbol — Train | 80.9% | 65.9% | 58.1% | 61.8% |
| Árbol — CV media | 79.5% | — | — | — |

**Variación CV (σ): 0.014** → modelo estable

### Análisis de overfitting/underfitting
- Diferencia Train–Test = **4.2%** → por debajo del umbral de 5%, sin overfitting significativo
- CV consistente con Test → buena generalización
- ✅ **Modelo EQUILIBRADO**

### Matriz de confusión (Test)

|  | Predicho: No canceló | Predicho: Canceló |
|---|---|---|
| **Real: No canceló** | 880 ✅ | 153 ❌ |
| **Real: Canceló** | 175 ❌ | 199 ✅ |

El principal punto débil es el **Recall (53%)**: el modelo no detecta ~175 clientes que sí cancelarán.

---

## 📈 Variables Más Relevantes

Ordenadas por correlación absoluta con `Abandonó_Servicio`:

| Variable | Correlación | Observación |
|---|---|---|
| Antigüedad | 0.354 | La más influyente — clientes nuevos cancelan mucho más |
| Cargo Total | 0.199 | Menor acumulado = mayor riesgo |
| Cargo Mensual | 0.193 | Tarifas altas sin valor percibido |
| Cargo Diario | 0.192 | Refleja el patrón del cargo mensual |
| Facturación Sin Papel | 0.171 | Perfil digital = más movilidad |
| Soporte Técnico | 0.165 | Su ausencia aumenta la fuga |
| Seguridad Online | 0.164 | Sin servicios extras, menos retención |
| Cliente Señor | 0.151 | Adultos mayores cancelan menos |

---

## 🎯 Conclusiones y Estrategias de Retención

### Conclusión del modelo
El Árbol de Decisión con `max_depth=5` supera la línea base en **+3.3 puntos** con buena estabilidad. Para mejorar el Recall se recomienda explorar **Random Forest** o **Gradient Boosting**, o ajustar el umbral de decisión para priorizar la detección de cancelaciones.

### Estrategias propuestas

**1. Onboarding reforzado (primeros 18 meses)**
- Descuentos progresivos durante los primeros 18 meses
- Check-in automático al mes 3, 6 y 12

**2. Incentivo a contratos de mayor plazo**
- 10–15% de descuento por contrato anual o bienal
- Paquetes bundle (internet + soporte + seguridad) con precio diferenciado

**3. Activación de servicios de valor añadido**
- Trial gratuito de 3 meses de Seguridad Online y Soporte Técnico
- Notificaciones proactivas de ahorro y uso para reforzar percepción de valor

**4. Intervención predictiva**
- Usar el modelo mensualmente para identificar clientes con probabilidad de cancelación > 60%
- Activar flujo de retención personalizado antes de que se vayan

---

*Proyecto desarrollado como parte del Challenge Data Science LATAM — Alura*
