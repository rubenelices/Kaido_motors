"""Rutas relativas a esta carpeta y lectura que conserva el código de producto NA."""
from pathlib import Path
import pandas as pd

PROYECTO = Path(__file__).resolve().parents[1]
DATOS = PROYECTO / 'data'
RESULTADOS = PROYECTO / 'resultados'

def leer_csv(nombre):
    ruta = DATOS / nombre
    if not ruta.is_file():
        raise FileNotFoundError(f'No se encuentra el CSV original: {ruta}')
    return pd.read_csv(ruta, keep_default_na=False, na_values=[''])
