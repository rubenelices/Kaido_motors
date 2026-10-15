"""Ejecuta los cuatro notebooks en orden y con kernels independientes.

Desde la raíz de esta carpeta: python -m src.ejecutar_notebooks
Para repetir uno: python -m src.ejecutar_notebooks 04_costes_y_politicas.ipynb
No crea un ZIP ni sobrescribe los documentos.
"""
from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from .rutas import PROYECTO, DATOS, RESULTADOS

NOTEBOOKS = ['01_datos_y_demanda_historica.ipynb', '02_modelo_y_validacion.ipynb',
             '03_escenarios_produccion_y_simulacion.ipynb', '04_costes_y_politicas.ipynb']


def huellas_datos():
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(DATOS.glob('*.csv'))}


def ejecutar(nombres=None):
    nombres = list(nombres or NOTEBOOKS)
    if any(n not in NOTEBOOKS for n in nombres):
        raise ValueError('Indica uno o varios nombres de los cuatro notebooks de esta entrega.')
    antes = huellas_datos()
    if len(antes) != 7:
        raise ValueError('La carpeta data debe contener los siete CSV originales.')
    registros = []
    for nombre in nombres:
        inicio = time.monotonic()
        ruta = PROYECTO / 'notebooks' / nombre
        nb = nbformat.read(ruta, as_version=4)
        nbformat.validate(nb)
        print(f'Ejecutando {nombre}...', flush=True)
        with tempfile.TemporaryDirectory(prefix='kaido-entrega-kernel-') as temporal:
            entorno = os.environ.copy()
            entorno.update(IPYTHONDIR=str(Path(temporal) / 'ipython'),
                           MPLCONFIGDIR=str(Path(temporal) / 'matplotlib'),
                           PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable, '-c', 'import matplotlib.font_manager'],
                           env=entorno, check=True, capture_output=True)
            km = KernelManager(kernel_name='python3', connection_file=str(Path(temporal) / 'kernel.json'))
            km.kernel_spec.argv[0] = sys.executable
            cliente = NotebookClient(nb, km=km, timeout=600, allow_errors=False,
                resources={'metadata': {'path': str(ruta.parent)}})
            try:
                cliente.execute(cwd=str(ruta.parent), env=entorno)
            finally:
                if km.has_kernel:
                    km.shutdown_kernel(now=True)
        nbformat.validate(nb)
        if any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs):
            raise RuntimeError(f'{nombre} contiene errores.')
        nbformat.write(nb, ruta)
        registro = {'notebook': nombre, 'celdas_codigo': sum(c.cell_type == 'code' for c in nb.cells),
                    'segundos': round(time.monotonic() - inicio, 2),
                    'sha256': hashlib.sha256(ruta.read_bytes()).hexdigest(), 'estado': 'Correcto'}
        registros.append(registro)
        print(f'Correcto: {nombre} ({registro["segundos"]} s).', flush=True)
    if huellas_datos() != antes:
        raise RuntimeError('Los datos originales han cambiado durante la ejecución.')
    RESULTADOS.mkdir(exist_ok=True)
    archivo_informe = RESULTADOS / 'verificacion_notebooks.json'
    anteriores = json.loads(archivo_informe.read_text()).get('notebooks', []) if archivo_informe.exists() else []
    nuevos = {r['notebook'] for r in registros}
    conservados = [r for r in anteriores if r['notebook'] not in nuevos
                   and (PROYECTO / 'notebooks' / r['notebook']).is_file()
                   and hashlib.sha256((PROYECTO / 'notebooks' / r['notebook']).read_bytes()).hexdigest() == r['sha256']]
    informe = {'python': sys.version.split()[0], 'notebooks': sorted(conservados + registros, key=lambda r: r['notebook']),
               'fuentes_sin_cambios': True, 'fuentes_sha256': antes,
               'dependencias': {n: version(n) for n in ['numpy', 'pandas', 'scipy',
                    'scikit-learn', 'matplotlib', 'nbformat', 'nbclient', 'ipykernel']}}
    archivo_informe.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + '\n')
    return registros


if __name__ == '__main__':
    ejecutar(sys.argv[1:] or None)
