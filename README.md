# TP3 Análisis Predictivo (ITBA): ¿nuevo o usado?

Examen 3 de Análisis Predictivo (Licenciatura en Analítica, ITBA): competencia de Kaggle **ap-2026-q-2**. Hay que clasificar si cada publicación de MercadoLibre es un producto nuevo (`new`) o usado (`used`). La métrica es Accuracy.

**Resultado:** 0,93788 en el leaderboard privado, 0,93742 en el leaderboard público y 93,54 % de accuracy en validación cruzada (5 folds estratificados). El modelo final es un LightGBM sobre 125 features:
- precio y cantidades;
- columnas anidadas aplanadas;
- frases clave del título y la garantía;
- target encoding de vendedor y categoría;
- probabilidad de "usado" según el título (TF-IDF + regresión logística);
- perfil del vendedor;
- tiempo entre la creación y la última actualización.

## Contenido

| Archivo | Qué es |
|---|---|
| `presentacion/TP3_Grinberg_Nuevo_o_Usado.pdf` y `.pptx` | **Entregable a):** la presentación. |
| `entrega/01_entrenamiento.ipynb` | **Entregable b):** lee train, arma las features, ajusta los codificadores con todo train, entrena LightGBM y guarda todo en `entrega/modelo/`. |
| `entrega/02_prediccion.ipynb` | **Entregable c):** carga `entrega/modelo/` (no ajusta nada), aplica las transformaciones a test, predice y guarda `entrega/submission.csv`. |
| `entrega/features.py` | Features y codificadores del modelo, con comentarios. Lo usan los dos notebooks. |
| `entrega/modelo/` | Lo que genera el 01: codificadores ajustados (`codificador.joblib`), modelo (`lightgbm.joblib`) y orden de columnas (`columnas.json`). Está incluido para poder correr el 02 directamente. |
| `entrega/submission.csv` | El archivo enviado a Kaggle (`ID,condition`, 30.000 filas). |
| `entrega/requirements.txt` | Librerías con versión fija (Python 3.14). |

## Cómo correrlo

1. Instalar las librerías:
   ```
   pip install -r entrega/requirements.txt
   ```
2. Bajar `train_data.jsonlines` y `test_data.jsonlines` de la competencia y ponerlos en una carpeta `data/` en la raíz del repo. Los datos no se suben al repo por las reglas de Kaggle.
   ```
   data/
     train_data.jsonlines
     test_data.jsonlines
   entrega/
   presentacion/
   ```
3. Abrir los notebooks desde la carpeta `entrega/` y correr `01_entrenamiento.ipynb` (unos 2 minutos) y después `02_prediccion.ipynb` (alrededor de 1 minuto). Para usar el modelo ya guardado, alcanza con correr el 02.

Todo tiene semilla fija: `entrega/submission.csv` sale idéntico al enviado a Kaggle.
