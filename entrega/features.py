"""Features del modelo final (submission 05): publicaciones de MercadoLibre, nuevo o usado.

Hay dos tipos de pasos:
1. `construir_features(df)`: transformaciones **fila por fila** sobre los datos crudos (train o test tal
   como salen de `pd.read_json(..., lines=True)`). No aprenden nada de los datos, así que se pueden
   aplicar igual a train y a test.
2. `crear_codificador()`: lo que **se aprende de los datos** (target encoding, frecuencias, TF-IDF del
   título, perfil del vendedor y de la categoría). Se ajusta solo con train y después se aplica a test.
   En train, cada fila se codifica con un cross-fitting interno de 5 partes: nunca ve su propio target.

No se imputan nulos, no se recortan outliers y no se aplica log: LightGBM es un modelo de árboles y
solo usa el orden de los valores. Los nulos (por ejemplo, un vendedor que no está en train) quedan
como faltantes y LightGBM los maneja solo.
"""
import unicodedata

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import TargetEncoder

# Suavizado del target encoding: un grupo con pocas publicaciones queda cerca del promedio general.
SUAVIZADO = 10

# Valores de las categóricas en train. Se fijan para que train y test tengan las mismas columnas.
LISTING_TYPES = ["bronze", "free", "gold", "gold_premium", "gold_pro", "gold_special", "silver"]
STATUSES = ["active", "closed", "not_yet_active", "paused"]
SHIPPING_MODES = ["custom", "me1", "me2", "not_specified"]
MEDIOS_PAGO = ["Efectivo", "Transferencia bancaria", "Tarjeta de crédito", "Acordar con el comprador",
               "Giro postal", "MercadoPago", "Visa", "MasterCard", "Visa Electron", "Mastercard Maestro",
               "Contra reembolso", "American Express", "Diners", "Cheque certificado"]
TAGS = ["dragged_bids_and_visits", "good_quality_thumbnail", "dragged_visits", "free_relist",
        "poor_quality_thumbnail"]

# Frases clave (texto normalizado: minúsculas y sin tildes; \b = palabra completa).
# En el EDA, "usado", "como nuevo", "impecable" o "funcionando" en el título son casi siempre usados;
# "nuevo", "stock" o "sellado", casi siempre nuevos.
NUEVO = r"\bnuev[oa]s?\b"
COMO_NUEVO = r"\bcomo nuev[oa]s?\b"
FRASES_TITULO = {
    "usado": r"\busad[oa]s?\b",
    "como_nuevo": COMO_NUEVO,
    "impecable": r"\bimpecables?\b",
    "funcionando": r"\bfuncionando\b",
    "antiguo": r"\bantigu[oa]s?\b",
    "stock": r"\bstock\b",
    "nuevo": NUEVO,
    "sellado": r"\bsellad[oa]s?\b",
    "garantia": r"\bgarantia\b",
    "fabrica": r"\bfabrica\b",
    "original": r"\boriginal(?:es)?\b",
    "caja": r"\bcaja\b",
}
FRASES_GARANTIA = {
    "sellado": r"\bsellad[oa]s?\b",
    "fabrica": r"\bfabrica\b",
    "caja": r"\bcaja\b",
    "nuevo": NUEVO,
    "sin_garantia": r"\bsin garantia\b",
    "meses": r"\bmeses\b",
    "calificaciones": r"\bcalificaciones\b",
    "reputacion": r"\breputacion\b",
    "oferta": r"\boferta\b",
}
# Frases agregadas en el exp. 19, buscadas en el título y en la garantía. "usado" en el título ya está
# en FRASES_TITULO, así que esa solo se busca en la garantía.
FRASES_NUEVAS = {
    "refabricado": r"\brefabricad[oa]s?\b",
    "reacondicionado": r"\breacondicionad[oa]s?\b",
    "outlet": r"\boutlet\b",
    "liquido": r"\bliquid(?:o|amos|acion)\b",
    "lote": r"\blotes?\b",
    "coleccion": r"\bcoleccion(?:es)?\b|\bcoleccionables?\b",
    "edicion": r"\bedicion(?:es)?\b",
    "a_estrenar": r"\ba estrenar\b",
    "usado": r"\busad[oa]s?\b",
    "restaurado": r"\brestaurad[oa]s?\b",
    "vintage": r"\bvintage\b",
    "sin_uso": r"\bsin uso\b",
    "libro": r"\blibros?\b",
    "vinilo": r"\bvinilos?\b|\blp\b",
    "revista": r"\brevistas?\b",
}

COLUMNAS_PERFIL = ["perfil_vendedor", "perfil_categoria", "perfil_free", "perfil_cantidad",
                   "perfil_garantia", "perfil_precio", "perfil_envio_gratis"]
# Columnas de los bloques agregados después del exp. 9. Van al final, en este orden, para reproducir
# exactamente el modelo de la 05 (`colsample_bytree` sortea columnas según su posición).
COLUMNAS_FINALES = ["te_vend_tipo", "te_vend_cat", "hora_inicio", "dia_semana_inicio", "dias_hasta_actualizacion"]


# ---------------------------------------------------------------------------------------------------
# 1. Features fila por fila
# ---------------------------------------------------------------------------------------------------

def normalizar(texto):
    """Minúsculas y sin tildes. Los nulos quedan como texto vacío."""
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def _tiene_texto(serie):
    return serie.map(lambda t: isinstance(t, str) and t.strip() != "")


def _nombre_columna(texto):
    return "".join(c if c.isalnum() else "_" for c in normalizar(texto))


def _one_hot(serie, valores, prefijo):
    return pd.DataFrame({f"{prefijo}_{v}": (serie == v).astype(int) for v in valores}, index=serie.index)


def features_base(df):
    """Precio, cantidades, dos booleanas y tipo de publicación y estado en one-hot.

    Las cantidades son la señal más fuerte del EDA: con más de 1 unidad, solo el 5 % es usado.
    El tipo `free` es 93 % usado y `gold_special`, 98 % nuevo. `buying_mode` no se usa: `classified`
    equivale a `accepts_mercadopago = False`.
    """
    X = df[["price", "initial_quantity", "available_quantity", "sold_quantity"]].astype(float)
    X = X.assign(accepts_mercadopago=df["accepts_mercadopago"].astype(int),
                 automatic_relist=df["automatic_relist"].astype(int))
    return pd.concat([X, _one_hot(df["listing_type_id"], LISTING_TYPES, "listing_type"),
                      _one_hot(df["status"], STATUSES, "status")], axis=1)


def features_indicadores(df):
    """Aplana las columnas anidadas: indicadores "tiene X", conteos, medios de pago, tags y envío.

    Los nulos son información (tener precio original, deal o tienda oficial es casi siempre nuevo;
    no tener fotos, 68 % usado), así que no se imputan: se convierten en 0/1 o en conteos.
    """
    medios = df["non_mercado_pago_payment_methods"].map(lambda v: {e["description"] for e in v})
    envio = df["shipping"]
    X = pd.DataFrame({
        "tiene_garantia": _tiene_texto(df["warranty"]),
        "tiene_tienda_oficial": df["official_store_id"].notna(),
        "tiene_video": df["video_id"].map(lambda v: isinstance(v, str) and v != ""),
        "tiene_deal": df["deal_ids"].map(len) > 0,
        "tiene_precio_original": df["original_price"].notna(),
        "tiene_descripcion": df["descriptions"].map(len) > 0,
        "n_fotos": df["pictures"].map(len),
        "n_atributos": df["attributes"].map(len),
        "n_variaciones": df["variations"].map(len),
        "n_medios_pago": df["non_mercado_pago_payment_methods"].map(len),
        "envio_gratis": envio.map(lambda d: d["free_shipping"]),
        "retiro_en_persona": envio.map(lambda d: d["local_pick_up"]),
        "moneda_usd": df["currency_id"] == "USD",
        "precio_simbolico": df["price"] <= 1,             # precios de $1 o menos: 68 % usado
        "precio_ficticio": df["price"] >= 10_000_000,     # 11.111.111, 123.456.789, ...
        "stock_9999": (df["initial_quantity"] >= 9999) | (df["available_quantity"] >= 9999),
    }, index=df.index).astype(int)
    pagos = pd.DataFrame({f"pago_{_nombre_columna(m)}": medios.map(lambda s, m=m: m in s).astype(int)
                          for m in MEDIOS_PAGO}, index=df.index)
    tags = pd.DataFrame({f"tag_{t}": df["tags"].map(lambda v, t=t: t in v).astype(int) for t in TAGS},
                        index=df.index)
    modo_envio = _one_hot(envio.map(lambda d: d["mode"]), SHIPPING_MODES, "envio_modo")
    return pd.concat([X, pagos, tags, modo_envio], axis=1)


def _frases(textos, frases, prefijo):
    """0/1 por frase. "nuevo" no cuenta cuando forma parte de "como nuevo" (que es usado)."""
    sin_como_nuevo = textos.str.replace(COMO_NUEVO, " ", regex=True)
    return pd.DataFrame({
        f"{prefijo}_{nombre}": (sin_como_nuevo if patron == NUEVO else textos)
        .str.contains(patron, regex=True).astype(int)
        for nombre, patron in frases.items()
    }, index=textos.index)


def features_texto(df):
    """Frases clave del título y de la garantía, y largo del título."""
    titulo = df["title"].map(normalizar)
    garantia = df["warranty"].map(normalizar)
    largo = pd.DataFrame({"largo_titulo": df["title"].str.len()}, index=df.index)
    return pd.concat([_frases(titulo, FRASES_TITULO, "titulo"),
                      _frases(garantia, FRASES_GARANTIA, "garantia"), largo], axis=1)


def features_ids(df):
    """IDs de vendedor y categoría (dos copias: una para el target encoding y otra para la frecuencia).

    El 72 % de los vendedores con 5+ publicaciones vende una sola clase, y hay categorías casi puras.
    """
    return pd.DataFrame({
        "te_vendedor": df["seller_id"].astype(str),
        "te_categoria": df["category_id"].astype(str),
        "frec_vendedor": df["seller_id"].astype(str),
        "frec_categoria": df["category_id"].astype(str),
    }, index=df.index)


def features_tfidf(df):
    """Título normalizado, para el TF-IDF del codificador (`ProbaTexto`)."""
    return pd.DataFrame({"tfidf_titulo": df["title"].map(normalizar)}, index=df.index)


def features_perfil(df):
    """Valores por fila con los que `PerfilGrupo` arma el perfil del vendedor y de la categoría."""
    return pd.DataFrame({
        "perfil_vendedor": df["seller_id"].astype(str),
        "perfil_categoria": df["category_id"].astype(str),
        "perfil_free": (df["listing_type_id"] == "free").astype(float),
        "perfil_cantidad": df["initial_quantity"].astype(float),
        "perfil_garantia": _tiene_texto(df["warranty"]).astype(float),
        "perfil_precio": df["price"].astype(float),
        "perfil_envio_gratis": df["shipping"].map(lambda d: d["free_shipping"]).astype(float),
    }, index=df.index)


def features_extra(df):
    """Bloques agregados después del exp. 9 (exps. 17, 18 y 19).

    - Vendedor × tipo de publicación y vendedor × categoría (exp. 17): un mismo vendedor puede publicar
      usados como `free` y nuevos como `gold_special`. El codificador les hace target encoding.
    - Tiempo (exp. 18): hora y día de la semana de `start_time`, y días entre `date_created` y
      `last_updated` (esta última es la que aporta la mejora).
    - Frases nuevas (exp. 19): 0/1 de `FRASES_NUEVAS` en el título y en la garantía.
    """
    X = pd.DataFrame(index=df.index)
    vendedor = df["seller_id"].astype(str)
    X["grupo_vend_tipo"] = vendedor + "|" + df["listing_type_id"].astype(str)
    X["grupo_vend_cat"] = vendedor + "|" + df["category_id"].astype(str)
    X["hora_inicio"] = df["start_time"].dt.hour
    X["dia_semana_inicio"] = df["start_time"].dt.day_of_week
    creada = pd.to_datetime(df["date_created"], utc=True)
    X["dias_hasta_actualizacion"] = (pd.to_datetime(df["last_updated"], utc=True) - creada).dt.total_seconds() / 86400
    titulo, garantia = df["title"].map(normalizar), df["warranty"].map(normalizar)
    for nombre, patron in FRASES_NUEVAS.items():
        if nombre != "usado":
            X[f"frase_titulo_{nombre}"] = titulo.str.contains(patron, regex=True).astype(int)
        X[f"frase_garantia_{nombre}"] = garantia.str.contains(patron, regex=True).astype(int)
    return X


def construir_features(df):
    """Junta todos los bloques fila por fila. La salida es la entrada de `crear_codificador`."""
    bloques = [features_base(df), features_indicadores(df), features_texto(df), features_ids(df),
               features_tfidf(df), features_perfil(df), features_extra(df)]
    return pd.concat(bloques, axis=1)


# ---------------------------------------------------------------------------------------------------
# 2. Codificadores (se ajustan solo con train)
# ---------------------------------------------------------------------------------------------------

class Frecuencia(BaseEstimator, TransformerMixin):
    """Cuántas OTRAS publicaciones del mismo vendedor (o categoría) hay en train.

    En train (`fit_transform`) se resta 1 para no contar la propia fila; en test la fila no está en
    train, así que no se resta. Así la columna significa lo mismo en los dos. No vistos = 0.
    """

    def fit(self, X, y=None):
        self.columnas_ = list(X.columns)
        self.conteos_ = {c: X[c].value_counts() for c in self.columnas_}
        return self

    def transform(self, X):
        return pd.DataFrame({c: X[c].map(self.conteos_[c]).fillna(0).astype(float) for c in self.columnas_},
                            index=X.index)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X) - 1

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columnas_, dtype=object)


class ProbaTexto(BaseEstimator, TransformerMixin):
    """Probabilidad de `used` según el título: TF-IDF + regresión logística (la feature más importante).

    TF-IDF de palabras (1-2 gramas) y de caracteres (3-5 gramas, dentro de cada palabra): los
    caracteres captan variantes como "usadito" o errores de tipeo. El resultado es una sola columna,
    que LightGBM combina con el resto.
    - `fit_transform` (train): cross-fitting interno de 5 partes; cada fila recibe la probabilidad de una
      regresión que no vio su target. Además se guarda una regresión ajustada con todo train.
    - `transform` (test) usa esa regresión completa.
    Las 6 regresiones son independientes y se ajustan en paralelo (hilos).
    """

    def __init__(self, n_jobs=6):
        self.n_jobs = n_jobs

    def _matriz(self, textos):
        return hstack([self.palabras_.transform(textos), self.caracteres_.transform(textos)]).tocsr()

    @staticmethod
    def _regresion():
        return LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        proba = self.modelo_.predict_proba(self._matriz(X.iloc[:, 0]))[:, self.clase_]
        return pd.DataFrame({"proba_titulo": proba}, index=X.index)

    def fit_transform(self, X, y):
        # El vocabulario se arma con todos los títulos de train (no usa el target)
        textos = X.iloc[:, 0]
        self.palabras_ = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.caracteres_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        M = hstack([self.palabras_.fit_transform(textos), self.caracteres_.fit_transform(textos)]).tocsr()
        y = np.asarray(y)
        partes = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(M, y))
        conjuntos = [ajuste for ajuste, _ in partes] + [np.arange(M.shape[0])]
        modelos = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._regresion().fit)(M[idx], y[idx]) for idx in conjuntos)
        self.modelo_ = modelos[-1]
        self.clase_ = list(self.modelo_.classes_).index("used")
        proba = np.empty(M.shape[0])
        for (_, aplicar), modelo in zip(partes, modelos):
            proba[aplicar] = modelo.predict_proba(M[aplicar])[:, self.clase_]
        return pd.DataFrame({"proba_titulo": proba}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(["proba_titulo"], dtype=object)


class PerfilGrupo(BaseEstimator, TransformerMixin):
    """Perfil del vendedor y de la categoría. No usa el target.

    Vendedor: % de publicaciones `free`, mediana de unidades, % con garantía, cantidad de categorías,
    mediana de precio y % con envío gratis (describe "qué tipo de vendedor es"). Categoría: mediana de
    precio y precio del ítem relativo a esa mediana (un usado suele ser más barato que lo típico).
    - `transform` (test): vendedor o categoría que no está en train = faltante.
    - `fit_transform` (train): cross-fitting de 5 partes; la fila no entra en su propio perfil y, como en
      test, hay vendedores no vistos.
    """

    VENDEDOR = ["vend_pct_free", "vend_mediana_cantidad", "vend_pct_garantia", "vend_n_categorias",
                "vend_mediana_precio", "vend_pct_envio_gratis"]
    CATEGORIA = ["cat_mediana_precio", "precio_relativo_cat"]

    def fit(self, X, y=None):
        self.vendedor_ = X.groupby("perfil_vendedor").agg(
            vend_pct_free=("perfil_free", "mean"),
            vend_mediana_cantidad=("perfil_cantidad", "median"),
            vend_pct_garantia=("perfil_garantia", "mean"),
            vend_n_categorias=("perfil_categoria", "nunique"),
            vend_mediana_precio=("perfil_precio", "median"),
            vend_pct_envio_gratis=("perfil_envio_gratis", "mean"),
        )
        self.mediana_categoria_ = X.groupby("perfil_categoria")["perfil_precio"].median()
        return self

    def transform(self, X):
        P = self.vendedor_.reindex(X["perfil_vendedor"].values).set_index(X.index)
        P["cat_mediana_precio"] = X["perfil_categoria"].map(self.mediana_categoria_)
        # 1 = precio típico de la categoría. Si la mediana es 0 o falta, queda nulo.
        P["precio_relativo_cat"] = X["perfil_precio"] / P["cat_mediana_precio"].replace(0, np.nan)
        return P[self.VENDEDOR + self.CATEGORIA]

    def fit_transform(self, X, y=None):
        self.fit(X)
        partes = KFold(n_splits=5, shuffle=True, random_state=42)
        salida = [PerfilGrupo().fit(X.iloc[ajuste]).transform(X.iloc[aplicar])
                  for ajuste, aplicar in partes.split(X)]
        return pd.concat(salida).reindex(X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.VENDEDOR + self.CATEGORIA, dtype=object)


class GrupoTE(BaseEstimator, TransformerMixin):
    """Target encoding de una clave combinada (vendedor × tipo, vendedor × categoría).

    % de usados entre las publicaciones de train con la misma clave, suavizado hacia el promedio general:
        (usados + suavizado * %usado_general) / (n + suavizado)
    Clave que no está en train = faltante. En `fit_transform` (train) se calcula con cross-fitting
    interno de 5 partes: la fila nunca ve su propio target.
    """

    def __init__(self, suavizado=SUAVIZADO):
        self.suavizado = suavizado

    def _tabla(self, claves, y):
        """Guarda, por clave, cuántas publicaciones hay y cuántas son usadas, y el % de usados general."""
        self.prior_ = y.mean()
        grupos = pd.Series(y).groupby(np.asarray(claves)).agg(["size", "sum"])
        self.n_, self.usados_ = grupos["size"], grupos["sum"]
        return self

    def _te(self, claves):
        claves = pd.Series(np.asarray(claves))
        n = claves.map(self.n_).fillna(0).to_numpy(dtype=float)
        usados = claves.map(self.usados_).fillna(0).to_numpy(dtype=float)
        return np.where(n > 0, (usados + self.suavizado * self.prior_) / (n + self.suavizado), np.nan)

    def fit(self, X, y):
        self.nombre_ = X.columns[0].replace("grupo_", "")
        return self._tabla(X.iloc[:, 0], (np.asarray(y) == "used").astype(float))

    def transform(self, X):
        return pd.DataFrame({f"te_{self.nombre_}": self._te(X.iloc[:, 0])}, index=X.index)

    def fit_transform(self, X, y):
        self.fit(X, y)
        claves, y_bin = X.iloc[:, 0].to_numpy(), (np.asarray(y) == "used").astype(float)
        te = np.empty(len(claves))
        for ajuste, aplicar in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(claves, y_bin):
            te[aplicar] = GrupoTE(self.suavizado)._tabla(claves[ajuste], y_bin[ajuste])._te(claves[aplicar])
        return pd.DataFrame({f"te_{self.nombre_}": te}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"te_{self.nombre_}"], dtype=object)


def crear_codificador():
    """Todo lo que se aprende de train, sobre la salida de `construir_features`. El resto de las columnas
    pasa igual.

    - Target encoding de vendedor y categoría (% de usados del grupo, suavizado; cross-fitting interno).
    - Frecuencia de vendedor y categoría (cuántas otras publicaciones tiene en train).
    - Probabilidad de usado según el título (`ProbaTexto`).
    - Perfil del vendedor y de la categoría (`PerfilGrupo`).
    - Target encoding de vendedor × tipo y vendedor × categoría (`GrupoTE`).
    """
    folds_internos = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pasos = [
        ("target_encoding", TargetEncoder(target_type="binary", smooth=SUAVIZADO, cv=folds_internos),
         ["te_vendedor", "te_categoria"]),
        ("frecuencia", Frecuencia(), ["frec_vendedor", "frec_categoria"]),
        ("proba_titulo", ProbaTexto(), ["tfidf_titulo"]),
        ("perfil", PerfilGrupo(), COLUMNAS_PERFIL),
        ("vend_tipo", GrupoTE(), ["grupo_vend_tipo"]),
        ("vend_cat", GrupoTE(), ["grupo_vend_cat"]),
    ]
    return ColumnTransformer(pasos, remainder="passthrough",
                             verbose_feature_names_out=False).set_output(transform="pandas")


def orden_columnas(columnas):
    """Orden de columnas del modelo de la 05: primero las del exp. 9 y al final los bloques extra.

    Las frases nuevas ya salen al final del codificador; solo hay que mover las de `COLUMNAS_FINALES`.
    """
    frases = [c for c in columnas if c.startswith("frase_")]
    resto = [c for c in columnas if c not in COLUMNAS_FINALES and c not in frases]
    return resto + COLUMNAS_FINALES + frases
