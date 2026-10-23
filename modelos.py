"""
Modelos del recomendador de huariques (Fase 6).

Las funciones se trasladan del notebook de modelado de la Fase 5
("Recomendador_Huariques_vf.ipynb") con la misma lógica, para que la
app recomiende exactamente lo que se evaluó.

>>> Para cambiar de modelo o hiperparámetros, solo se edita la CONFIGURACIÓN de abajo. <<<
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIGURACIÓN (decisión final de la Fase 5: notebook Recomendador_Huariques_vf)
# =========================================================
MODELO_HISTORIAL = "EASE"            # modelo para usuarios con historial
HIPERPARAMETROS = {                  # mejores valores en validación (Fase 5)
    "Popularidad":  {"m": 20},
    "Híbrido":      {"w": 0.1},
    "kNN encuesta": {"n_vecinos": 30},
    "EASE":         {"lam": 1000},
    "FunkSVD":      {"k": 5},
}
N_VECINOS_COLD_START = 30            # vecinos para el usuario nuevo (kNN por encuesta)
UMBRAL_RELEVANTE = 4                 # calificación ≥ 4 = "le gustó"

ARCHIVO_DATOS = Path(__file__).parent / "data" / "Huariques_listo_para_modelar.xlsx"

# Categoría del huarique -> columna de la encuesta con esa preferencia
MAPA_CATEGORIA = {"Criolla": "comida_criolla", "Marina/ceviche": "comida_marina_ceviche",
                  "Chifa": "comida_chifa", "Parrilla/anticuchos": "comida_parrilla_anticuchos",
                  "Sándwiches/snacks": "comida_sandwiches_snacks",
                  "Postres/dulces": "comida_postres_dulces"}


# =========================================================
# DATOS Y CONTEXTO
# =========================================================
@dataclass
class Contexto:
    """Tablas y matrices auxiliares que comparten todos los modelos."""
    usuarios: pd.DataFrame
    huariques: pd.DataFrame
    interacciones: pd.DataFrame
    N_U: int
    N_I: int
    DECL: np.ndarray          # usuario x huarique: 1 si declaró preferir su tipo de comida
    PERFIL: np.ndarray        # encuesta estandarizada (+ género en 0/1)
    cols_perfil: list
    perfil_media: np.ndarray
    perfil_desv: np.ndarray
    generos: list


def cargar_contexto(ruta=ARCHIVO_DATOS):
    usuarios = pd.read_excel(ruta, sheet_name="Usuarios").sort_values("user_idx").reset_index(drop=True)
    huariques = pd.read_excel(ruta, sheet_name="Huariques").sort_values("item_idx").reset_index(drop=True)
    interacciones = pd.read_excel(ruta, sheet_name="Interacciones")

    DECL = np.stack([usuarios[MAPA_CATEGORIA[c]].to_numpy() for c in huariques["categoria_estimada"]],
                    axis=1).astype(float)

    # Mismas columnas y estandarización que el notebook (kNN por encuesta)
    cols = [c for c in usuarios.columns if c.startswith(("val_", "comida_", "ocasion_"))
            and c != "comida_otro_texto"] + ["edad_ord", "frecuencia_ord", "gasto_ord", "novedad_ord"]
    P = usuarios[cols].to_numpy(float)
    media, desv = P.mean(0), P.std(0)
    generos = sorted(usuarios["genero"].unique())
    G = pd.get_dummies(usuarios["genero"]).reindex(columns=generos, fill_value=0).to_numpy(float)
    PERFIL = np.hstack([(P - media) / (desv + 1e-9), G])

    return Contexto(usuarios, huariques, interacciones, len(usuarios), len(huariques),
                    DECL, PERFIL, cols, media, desv, generos)


def matriz(ctx, df, modo="r"):
    """Matriz usuario x huarique: 'r' = calificación 1–5; 'bin' = 1 si calificación ≥ 4."""
    M = np.zeros((ctx.N_U, ctx.N_I))
    v = df["rating"].to_numpy(float)
    M[df["user_idx"], df["item_idx"]] = (v >= UMBRAL_RELEVANTE).astype(float) if modo == "bin" else v
    return M


# =========================================================
# MODELOS (cada uno devuelve una matriz de puntajes usuarios x huariques)
# =========================================================
def _z(S):
    """Estandariza los puntajes por usuario (fila)."""
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def popularidad(ctx, df, m=5):
    """Media suavizada por huarique: (suma + m·media_global) / (n + m)."""
    M = matriz(ctx, df)
    n = (M > 0).sum(0)
    return np.tile((M.sum(0) + m * df["rating"].mean()) / (n + m), (ctx.N_U, 1))


def declarado(ctx, df):
    """1 si el tipo de comida del huarique fue marcado en la encuesta; desempate por popularidad."""
    return ctx.DECL + 1e-6 * popularidad(ctx, df)


def hibrido(ctx, df, w=0.2):
    """z(Popularidad) + w · z(Perfil declarado)."""
    return _z(popularidad(ctx, df)) + w * _z(declarado(ctx, df))


def knn_encuesta(ctx, df, n_vecinos=30):
    """Recomienda lo que les gustó a los usuarios con la encuesta más parecida."""
    M = matriz(ctx, df, "bin")
    s = cosine_similarity(ctx.PERFIL)
    np.fill_diagonal(s, 0)
    s[:, (matriz(ctx, df) > 0).sum(1) == 0] = -1          # vecinos sin historial no aportan
    S = np.zeros_like(M)
    for u in range(ctx.N_U):
        v = np.argsort(-s[u])[:int(n_vecinos)]
        S[u] = np.maximum(s[u, v], 0) @ M[v]
    return S


def ease(ctx, df, lam=100):
    """EASE: B = I − P/diag(P), con P = (XᵀX + λI)⁻¹ y diag(B) = 0."""
    X = matriz(ctx, df, "bin")
    P = np.linalg.inv(X.T @ X + lam * np.eye(ctx.N_I))
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0)
    return X @ B


def funk_svd(ctx, df, k=5, lr=0.01, reg=0.05, epocas=60):
    """Calificación = μ + sesgo_usuario + sesgo_huarique + factores_u · factores_i (SGD)."""
    rng = np.random.default_rng(42)
    k = int(k)
    u, i, r = df["user_idx"].to_numpy(), df["item_idx"].to_numpy(), df["rating"].to_numpy(float)
    mu = r.mean()
    bu, bi = np.zeros(ctx.N_U), np.zeros(ctx.N_I)
    X, Y = rng.normal(0, .1, (ctx.N_U, k)), rng.normal(0, .1, (ctx.N_I, k))
    for _ in range(epocas):
        for t in rng.permutation(len(r)):
            a, c = u[t], i[t]
            e = r[t] - (mu + bu[a] + bi[c] + X[a] @ Y[c])
            xa = X[a].copy()
            bu[a] += lr * (e - reg * bu[a]); bi[c] += lr * (e - reg * bi[c])
            X[a] += lr * (e * Y[c] - reg * X[a]); Y[c] += lr * (e * xa - reg * Y[c])
    return mu + bu[:, None] + bi[None, :] + X @ Y.T


MODELOS = {"Popularidad": popularidad, "Híbrido": hibrido, "kNN encuesta": knn_encuesta,
           "EASE": ease, "FunkSVD": funk_svd}

DESCRIPCION = {
    "Popularidad": "Huariques mejor valorados en general (media suavizada); no personaliza.",
    "Híbrido": "Popularidad + coincidencia con los tipos de comida que el usuario declaró en la encuesta.",
    "kNN encuesta": "Lo que les gustó a los usuarios con respuestas de encuesta más parecidas.",
    "EASE": "Relaciones huarique–huarique aprendidas de quién aprobó qué (innovación, solución cerrada).",
    "FunkSVD": "Factores latentes y sesgos que predicen la calificación 1–5.",
}


def puntajes(ctx, df, nombre):
    """Matriz de puntajes del modelo `nombre` con sus hiperparámetros configurados."""
    return MODELOS[nombre](ctx, df, **HIPERPARAMETROS[nombre])


# =========================================================
# USUARIO NUEVO (COLD START): kNN POR ENCUESTA
# =========================================================
def vector_perfil(ctx, respuestas, genero):
    """Convierte las respuestas del formulario en el mismo vector estandarizado de PERFIL."""
    x = np.array([float(respuestas.get(c, 0)) for c in ctx.cols_perfil])
    g = np.array([1.0 if genero == gen else 0.0 for gen in ctx.generos])
    return np.hstack([(x - ctx.perfil_media) / (ctx.perfil_desv + 1e-9), g])


def knn_usuario_nuevo(ctx, df, vec, n_vecinos=N_VECINOS_COLD_START):
    """Puntaje por huarique = suma de similitudes de los vecinos a los que les gustó (≥ 4)."""
    M = matriz(ctx, df, "bin")
    s = cosine_similarity(vec.reshape(1, -1), ctx.PERFIL).ravel()
    s[(matriz(ctx, df) > 0).sum(1) == 0] = -1
    v = np.argsort(-s)[:int(n_vecinos)]
    return np.maximum(s[v], 0) @ M[v]


# =========================================================
# RANKING TOP-N
# =========================================================
def top_n(scores, n, excluir=(), candidatos=None, desempate=None):
    """
    Índices de los n huariques con mayor puntaje.
    - excluir: huariques ya calificados por el usuario (nunca se recomiendan).
    - candidatos: máscara booleana de los huariques que pasan los filtros.
    - desempate: puntaje secundario (popularidad) para ordenar empates, p. ej. puntaje 0.
    """
    scores = np.asarray(scores, float).ravel()
    validos = np.ones(len(scores), bool) if candidatos is None else np.asarray(candidatos, bool).copy()
    validos[list(excluir)] = False
    sec = np.zeros(len(scores)) if desempate is None else np.asarray(desempate, float).ravel()
    orden = np.lexsort((-sec, -scores))            # primero el puntaje, luego el desempate
    return [int(j) for j in orden if validos[j]][:n]
