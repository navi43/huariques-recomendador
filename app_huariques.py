"""
Huariques de Lima | Sistema de recomendación (Fase 6: prototipo)

Ejecutar desde esta carpeta:
    streamlit run app_huariques.py
"""
import numpy as np
import pandas as pd
import streamlit as st

import modelos as mo

st.set_page_config(page_title="Huariques | Recomendador", page_icon="🍲", layout="wide")


# =========================================================
# CARGA Y ENTRENAMIENTO (una sola vez; se guarda en caché)
# =========================================================
@st.cache_resource
def cargar():
    return mo.cargar_contexto()


@st.cache_resource
def entrenar(nombre):
    """Puntajes del modelo entrenado con todas las calificaciones disponibles."""
    ctx = cargar()
    return mo.puntajes(ctx, ctx.interacciones, nombre)


ctx = cargar()
huariques = ctx.huariques
inter = ctx.interacciones
# Popularidad suavizada: desempate y respaldo cuando un modelo no distingue huariques
pop = mo.popularidad(ctx, inter, **mo.HIPERPARAMETROS["Popularidad"])[0]

# Opciones del formulario (mismas preguntas y códigos de la encuesta original)
EDAD = {"18 - 24": 1, "25 - 34": 2, "35 - 44": 3, "45 - 54": 4, "55 +": 5}
FRECUENCIA = {"1 vez al mes": 1, "2-3 veces al mes": 2, "1 vez por semana": 3, "Más de 1 vez por semana": 4}
GASTO = {"Hasta S/20": 1, "S/21 - 40": 2, "S/41 - 70": 3, "Más de S/70": 4}
NOVEDAD = {"Siempre a lugares conocidos": 1, "Casi siempre a conocidos": 2,
           "Conocidos o nuevos por igual": 3, "Casi siempre a lugares nuevos": 4}
COMIDA = {"Criolla": "comida_criolla", "Marina / ceviche": "comida_marina_ceviche", "Chifa": "comida_chifa",
          "Nikkei": "comida_nikkei", "Parrilla / anticuchos": "comida_parrilla_anticuchos",
          "Sándwiches / snacks": "comida_sandwiches_snacks", "Postres / dulces": "comida_postres_dulces",
          "Andina": "comida_andina", "Amazónica": "comida_amazonica", "Otro": "comida_otro"}
VALORA = {"Sabor de la comida": "val_sabor_de_la_comida", "Precio": "val_precio",
          "Tamaño de las porciones": "val_tamano_de_las_porciones",
          "Rapidez de la atención": "val_rapidez_de_la_atencion", "Cercanía": "val_cercania",
          "Decoración / estética": "val_decoracion_estetica"}
OCASION = {"Almuerzo / cena": "ocasion_almuerzo_cena", "Salida con amigos": "ocasion_salida_con_amigos",
           "Salida en familia": "ocasion_salida_en_familia", "Pareja": "ocasion_pareja",
           "Antojo": "ocasion_antojo", "Celebración": "ocasion_celebracion"}


# =========================================================
# BARRA LATERAL: N Y FILTROS
# =========================================================
with st.sidebar:
    st.header("⚙️ Opciones")
    n = st.slider("Número de recomendaciones (N)", 3, 15, 5)
    st.subheader("Filtros")
    f_distrito = st.multiselect("Distrito del huarique", sorted(huariques["distrito"].unique()),
                                placeholder="Todos los distritos", select_all=False,
                                help="Puedes elegir varios (p. ej., distritos cercanos); vacío = todos.")
    f_categoria = st.multiselect("Tipo de comida", sorted(huariques["categoria_estimada"].unique()),
                                 placeholder="Todos los tipos", select_all=False,
                                 help="Puedes elegir varios; vacío = todos.")
    st.caption(f"Catálogo: {len(huariques)} huariques · {len(ctx.usuarios)} usuarios · "
               f"{len(inter):,} calificaciones".replace(",", "."))

candidatos = np.ones(len(huariques), bool)
if f_distrito:
    candidatos &= huariques["distrito"].isin(f_distrito).to_numpy()
if f_categoria:
    candidatos &= huariques["categoria_estimada"].isin(f_categoria).to_numpy()


def mostrar(ids, puntaje):
    """Tabla del Top-N con la información de cada huarique."""
    if not ids:
        st.warning("Ningún huarique cumple los filtros (o el usuario ya calificó todos los que los cumplen). "
                   "Prueba agregando más distritos o tipos de comida, o deja el filtro vacío.")
        return
    df = huariques.loc[ids, ["huarique", "distrito", "categoria_estimada", "rating_medio",
                             "n_calificaciones"]].reset_index(drop=True)
    df.insert(0, "Ranking", range(1, len(df) + 1))
    df["Puntaje del modelo"] = np.asarray(puntaje)[ids]
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "Ranking": st.column_config.NumberColumn("#", width="small"),
            "huarique": st.column_config.TextColumn("Huarique", width="medium"),
            "distrito": "Distrito", "categoria_estimada": "Tipo de comida",
            "rating_medio": st.column_config.NumberColumn("Calificación media", format="%.2f ⭐", width="small"),
            "n_calificaciones": st.column_config.NumberColumn("N.º calif.", width="small"),
            "Puntaje del modelo": st.column_config.NumberColumn(
                "Puntaje", format="%.3f", width="small",
                help="Sirve para ordenar dentro del modelo; no es una probabilidad ni se compara entre modelos."),
        })
    if len(ids) < n:
        st.info(f"Solo {len(ids)} huarique(s) cumplen los filtros y aún no han sido calificados por el usuario.")


# =========================================================
# PANTALLA PRINCIPAL
# =========================================================
st.title("🍲 Huariques de Lima | Sistema de recomendación")
modo = st.radio("Tipo de usuario", ["Usuario con historial", "Usuario nuevo"], horizontal=True)

if modo == "Usuario con historial":
    st.subheader("👤 Recomendación personalizada")
    c1, _ = st.columns([1, 2])
    with c1:
        usuario = st.selectbox("Usuario", ctx.usuarios["user_id"])
    modelo = mo.MODELO_HISTORIAL                     # decisión final de la Fase 5
    st.caption(f"**Modelo: {modelo}** (elegido en la Fase 5). {mo.DESCRIPCION[modelo]}")

    u = int(ctx.usuarios.loc[ctx.usuarios["user_id"] == usuario, "user_idx"].iloc[0])
    vistos = inter.loc[inter["user_idx"] == u, "item_idx"].tolist()
    S = entrenar(modelo)[u]
    ids = mo.top_n(S, n, excluir=vistos, candidatos=candidatos, desempate=pop)

    st.markdown(f"Este usuario calificó **{len(vistos)}** huarique(s); no se le vuelven a recomendar.")
    st.subheader(f"🎯 Top-{n} recomendados")
    mostrar(ids, S)

else:
    st.subheader("📝 Usuario nuevo: cuéntanos tus gustos")
    st.info("Todavía no tienes calificaciones, así que el sistema usa tus respuestas para encontrar "
            f"a los {mo.N_VECINOS_COLD_START} usuarios con gustos más parecidos y te recomienda "
            "los huariques que más les gustaron (calificación ≥ 4).")

    with st.form("encuesta"):
        c1, c2 = st.columns(2)
        with c1:
            comidas = st.multiselect("¿Qué tipos de comida prefieres?", list(COMIDA), placeholder="Elige una o varias")
            valora = st.multiselect("¿Qué valoras al elegir un huarique?", list(VALORA), placeholder="Elige una o varias")
            ocasion = st.multiselect("¿En qué ocasiones vas?", list(OCASION), placeholder="Elige una o varias")
        with c2:
            edad = st.selectbox("Edad", list(EDAD), index=1)
            genero = st.selectbox("Género", ctx.generos)
            frecuencia = st.selectbox("¿Con qué frecuencia vas a huariques?", list(FRECUENCIA), index=2)
            gasto = st.selectbox("¿Cuánto gastas por persona?", list(GASTO), index=1)
            novedad = st.selectbox("¿Prefieres lugares conocidos o nuevos?", list(NOVEDAD), index=2)
        enviado = st.form_submit_button("Recomendar huariques", type="primary")

    if enviado:
        respuestas = {COMIDA[c]: 1 for c in comidas} | {VALORA[v]: 1 for v in valora} \
                     | {OCASION[o]: 1 for o in ocasion}
        respuestas |= {"edad_ord": EDAD[edad], "frecuencia_ord": FRECUENCIA[frecuencia],
                       "gasto_ord": GASTO[gasto], "novedad_ord": NOVEDAD[novedad]}
        # Se guarda en la sesión para que cambiar N o los filtros no borre el resultado
        st.session_state["respuestas_nuevo"] = (respuestas, genero, bool(comidas))

    if "respuestas_nuevo" in st.session_state:
        respuestas, genero, marco_comida = st.session_state["respuestas_nuevo"]
        vec = mo.vector_perfil(ctx, respuestas, genero)
        S = mo.knn_usuario_nuevo(ctx, inter, vec)
        ids = mo.top_n(S, n, candidatos=candidatos, desempate=pop)

        st.subheader(f"🎯 Top-{n} recomendados para ti")
        mostrar(ids, S)
        respaldo = sum(S[j] == 0 for j in ids)
        if respaldo:
            st.caption(f"{respaldo} huarique(s) de la lista no les gustó a ninguno de tus vecinos: se completaron "
                       "con los mejor valorados en general (popularidad suavizada) como respaldo.")
        if not marco_comida:
            st.caption("Consejo: marca al menos un tipo de comida para recomendaciones más personalizadas.")

st.divider()
st.caption("Prototipo académico · Ciencia de Datos I (UNALM). Datos de una encuesta propia: 30 usuarios reales "
           "y 178 sintéticos. Los modelos se entrenan con todas las calificaciones disponibles.")
