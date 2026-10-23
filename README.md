# Huariques de Lima: prototipo del sistema de recomendación (Fase 6)

Aplicación en **Streamlit** que genera recomendaciones Top-N de huariques de Lima con dos modos:

| Modo | Para quién | Cómo recomienda |
|---|---|---|
| **Usuario con historial** | Cualquiera de los 208 usuarios del dataset | **EASE**, el modelo elegido en la Fase 5 (configurado en `modelos.py`). |
| **Usuario nuevo (cold start)** | Una persona sin calificaciones | Llena un formulario con las preguntas de la encuesta; **kNN por encuesta** busca los 30 usuarios con perfil más parecido y recomienda lo que les gustó (calificación ≥ 4). |

En los dos modos se elige el tamaño de la lista (N = 3 a 15) y se puede **filtrar por distrito y tipo de comida**: se pueden elegir varios (por ejemplo, distritos cercanos), y cada opción muestra cuántos huariques tiene. Nunca se recomienda un huarique que el usuario ya calificó.

## Estructura

```
03_App/
├── app_huariques.py     # interfaz Streamlit
├── modelos.py           # modelos, configuración y lógica de ranking
├── preparar_datos.py    # genera data/ a partir de 01_Datos con solo las columnas necesarias
├── data/
│   └── Huariques_listo_para_modelar.xlsx   # versión reducida de la salida del notebook 01
├── requirements.txt
└── README.md
```

## Cómo ejecutarla en local

Requisitos: Python 3.10 o superior.

```bash
cd PC2/03_App
pip install -r requirements.txt
streamlit run app_huariques.py
```

Se abre en el navegador en `http://localhost:8501`.

## Estrategia de cold start

Un usuario nuevo no tiene calificaciones, por eso los modelos colaborativos (EASE, FunkSVD) no pueden personalizarle nada. La app resuelve ese caso en dos niveles:

1. **kNN por encuesta.** Las respuestas del formulario (tipos de comida, qué valora, ocasiones, edad, frecuencia, gasto, preferencia por lugares nuevos y género) se convierten en el mismo vector estandarizado que se usó con los encuestados. Se calcula la similitud coseno con los 208 usuarios, se toman los 30 más parecidos y cada huarique recibe como puntaje la suma de similitudes de los vecinos que lo calificaron con 4 o 5.
2. **Respaldo por popularidad.** Si un huarique no le gustó a ningún vecino, su puntaje es 0. En ese caso, o cuando los filtros dejan pocos candidatos, el orden se completa con la **media suavizada** de calificaciones (Popularidad, m = 20). La app avisa cuando usa este respaldo.

## Coherencia con la Fase 5

`modelos.py` copia las funciones del notebook de modelado sin cambiar la lógica. Se comprobó con el Excel completo de `01_Datos` que, entrenando con `train` y evaluando en `validacion`, reproduce exactamente el NDCG@10 del notebook: Popularidad 0.389, Híbrido 0.399, kNN encuesta 0.361, EASE 0.350 y FunkSVD 0.339. En la app los modelos se entrenan con **todas** las calificaciones, porque ya no se está evaluando sino recomendando.

### Cambiar el modelo o los hiperparámetros

La configuración actual es la decisión final de la Fase 5 (`Recomendador_Huariques_vf.ipynb`): EASE para usuarios con historial y kNN por encuesta para usuarios nuevos. Para cambiarla, solo se edita el bloque `CONFIGURACIÓN` al inicio de `modelos.py`:

```python
MODELO_HISTORIAL = "EASE"            # modelo elegido en validación
HIPERPARAMETROS = {...}              # hiperparámetros elegidos en validación
N_VECINOS_COLD_START = 30
```

## Datos publicados (minimización)

Como el repositorio es público, `data/Huariques_listo_para_modelar.xlsx` es una **versión reducida** del archivo de `01_Datos`, y solo contiene lo que usan los modelos:
- **Usuarios:** ID anónimo, género y las respuestas de la encuesta codificadas (0/1 u ordinales).
- **Huariques:** el catálogo completo.
- **Interacciones:** usuario, huarique y calificación.

Se quitaron la marca temporal, el distrito de residencia, el rango de edad en texto, los textos libres, el origen real/sintético y la partición train/validación/test. Se verificó que las recomendaciones son idénticas a las obtenidas con el archivo completo. Si los datos cambian, se regenera con:

```bash
python preparar_datos.py
```

## Publicación en Streamlit Community Cloud

1. Subir el contenido de esta carpeta a un repositorio de GitHub (puede ser público).
2. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub → **Create app** → elegir el repositorio, la rama y el archivo principal `app_huariques.py` (o `PC2/03_App/app_huariques.py` si se sube todo el proyecto).
3. En **Advanced settings**, elegir Python 3.12.
4. Pulsar **Deploy** y copiar el enlace público para el informe.

## Limitaciones del prototipo

- El 85 % de las calificaciones viene de usuarios sintéticos, así que las recomendaciones reflejan en parte las reglas de la simulación.
- `categoria_estimada` se dedujo del nombre del huarique y puede tener errores.
- Nikkei, andina y amazónica se pueden marcar en el formulario, pero el catálogo no tiene huariques de esas categorías.
