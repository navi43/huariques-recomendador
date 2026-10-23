"""
Genera data/Huariques_listo_para_modelar.xlsx para la app a partir de la salida del notebook 01,
conservando solo las columnas que usan los modelos (minimización de datos para publicar el repositorio).

Se quitan, entre otras: marca temporal, distrito de residencia, rango de edad en texto, textos libres
(comida_otro_texto, recomendacion_texto), origen real/sintético y la partición train/validación/test.

Ejecutar desde la carpeta 03_App:
    python preparar_datos.py
"""
from pathlib import Path

import pandas as pd

ORIGEN = Path(__file__).parent.parent / "01_Datos" / "Huariques_listo_para_modelar.xlsx"
DESTINO = Path(__file__).parent / "data" / "Huariques_listo_para_modelar.xlsx"

usuarios = pd.read_excel(ORIGEN, sheet_name="Usuarios")
huariques = pd.read_excel(ORIGEN, sheet_name="Huariques")
interacciones = pd.read_excel(ORIGEN, sheet_name="Interacciones")

encuesta = [c for c in usuarios.columns if c.startswith(("val_", "comida_", "ocasion_")) and c != "comida_otro_texto"] \
           + ["edad_ord", "frecuencia_ord", "gasto_ord", "novedad_ord"]
usuarios = usuarios[["user_id", "user_idx", "genero"] + encuesta]
interacciones = interacciones[["user_id", "item_id", "user_idx", "item_idx", "rating"]]

DESTINO.parent.mkdir(exist_ok=True)
with pd.ExcelWriter(DESTINO) as xl:
    usuarios.to_excel(xl, sheet_name="Usuarios", index=False)
    huariques.to_excel(xl, sheet_name="Huariques", index=False)
    interacciones.to_excel(xl, sheet_name="Interacciones", index=False)
print(f"{DESTINO.name}: {usuarios.shape[1]} columnas de usuarios, {len(huariques)} huariques, "
      f"{len(interacciones)} calificaciones")
