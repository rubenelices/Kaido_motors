"""Etapa 07. Contribución de la marca y efectos de caja de la red, separados.

El bruto ya incorpora fabricación, transporte, margen del concesionario y
aduana base. No se deducen otra vez. El coste de oportunidad es informativo.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from .rutas import PROYECTO, RESULTADOS, leer_csv
from .prevision import SEGMENTOS, huellas
from .escenarios import ORDEN
from .produccion import cargar_politica, cargar_escenarios, sha_archivos


def cargar_costes():
    c = json.loads((PROYECTO / 'configuracion/costes.json').read_text())
    validar_costes(c)
    return c


def validar_costes(c):
    for k in ['coste_marca_eur_dia', 'tipo_credito_0_90', 'tipo_credito_91_180',
              'transfer_fraccion_bruto', 'csuv_vm_mas_180_factor']:
        if not np.isfinite(c[k]) or c[k] < 0:
            raise ValueError(f'Coste o tipo no válido: {k}')
    if c['dias_anio_financiero'] <= 0 or c['transfer_fraccion_bruto'] > 1:
        raise ValueError('Año financiero o coste de transfer no válido.')
    if c['base_arancel_cambio'] not in ['nuevas_y_transfers', 'todo_retail']:
        raise ValueError('Base de arancel/cambio desconocida.')


def cargar_trayectorias():
    """Comprueba fuentes, configuración y cadena de resultados antes de valorar."""
    _, datos04, _ = cargar_escenarios()
    d = RESULTADOS / '06_simulacion_mensual'
    m = json.loads((d / 'metodo.json').read_text())
    if m['fuentes_sha256'] != huellas() or m['politica'] != cargar_politica():
        raise ValueError('Datos o política cambiados: regenera producción y simulación en el notebook 03.')
    for carpeta, registro in [(d, m['salidas_sha256']),
            (RESULTADOS / '05_produccion_stock', m['etapa_05_sha256']),
            (RESULTADOS / '04_escenarios_incertidumbre', m['etapa_04_sha256'])]:
        for nombre, h in registro.items():
            if hashlib.sha256((carpeta / nombre).read_bytes()).hexdigest() != h:
                raise ValueError(f'Entrada desactualizada: {nombre}. Regenera el bloque de simulación del notebook 03.')
    with np.load(d / 'trayectorias_A013.npz', allow_pickle=False) as f:
        if f['escenarios'].tolist() != ORDEN or f['segmentos'].tolist() != SEGMENTOS:
            raise ValueError('Etiquetas de trayectorias incompatibles.')
        out = {k: f[k].copy() for k in f.files if k not in ['escenarios', 'segmentos', 'meses']}
    return out, datos04


def parametros_vm(fichas, c):
    z = fichas.set_index('segmento').loc[SEGMENTOS].copy()
    faltan = z.vm_real_mas_180_eur.isna()
    if faltan.sum() != 1 or not faltan.loc['CSUV']:
        raise ValueError('Las ausencias de VM han cambiado: revisa la imputación.')
    z.loc['CSUV', 'vm_real_mas_180_eur'] = z.loc['CSUV', 'vm_real_91_180_eur'] * c['csuv_vm_mas_180_factor']
    z['vm_mas_180_imputado'] = faltan
    return z.reset_index()


def valorar(out, fichas, entradas, c):
    """Valora cada réplica antes de agregar. Importes en euros, sin redondear."""
    validar_costes(c)
    z = parametros_vm(fichas, c).set_index('segmento').loc[SEGMENTOS]
    vector = lambda k: z[k].to_numpy(dtype=float)[None, None, None, :]
    shape = out['ventas'].shape
    if shape[0] != 3 or shape[2:] != (12, 7):
        raise ValueError('Se esperan tres escenarios, doce meses y siete segmentos.')
    if entradas.duplicated(['escenario', 'segmento']).any() or len(entradas) != 21:
        raise ValueError('Costes de escenario incompletos o duplicados.')
    def coste(k):
        return np.stack([entradas.query('escenario == @e').set_index('segmento').loc[SEGMENTOS, k]
                         .to_numpy(dtype=float) for e in ORDEN])[:, None, None, :]
    # FIFO consume antes los coches de A012. Mantienen su coste de importación ya incurrido.
    inicial = out['stock_inicio'][:, :, 0, :]
    anteriores = out['ventas_locales'].cumsum(axis=2) - out['ventas_locales']
    locales_iniciales = np.minimum(out['ventas_locales'], np.maximum(0, inicial[:, :, None, :] - anteriores))
    nuevas = out['ventas'] - locales_iniciales
    base_importacion = nuevas if c['base_arancel_cambio'] == 'nuevas_y_transfers' else out['ventas']
    e = {}
    e['ventas_stock_A012'] = locales_iniciales
    e['ventas_nuevas_y_transfers'] = nuevas
    e['beneficio_bruto_eur'] = out['ventas'] * vector('beneficio_bruto_marca_eur')
    e['vm_eur'] = sum(out[k] * vector(v) for k, v in [
        ('ventas_0_90', 'vm_real_0_90_eur'), ('ventas_91_180', 'vm_real_91_180_eur'),
        ('ventas_mas_180', 'vm_real_mas_180_eur')])
    e['vm_tarifa_eur'] = out['ventas'] * vector('vm_tarifa_eur')
    e['vm_exceso_vs_fresco_eur'] = e['vm_eur'] - out['ventas'] * vector('vm_real_0_90_eur')
    # Este exceso puede ser negativo: en B/SSUV el VM histórico no crece monótonamente.
    e['coste_stock_marca_eur'] = c['coste_marca_eur_dia'] * (out['dias_coche_0_90'] + out['dias_coche_91_180'])
    e['coste_transfers_eur'] = out['transfers'] * c['transfer_fraccion_bruto'] * vector('beneficio_bruto_marca_eur')
    e['arancel_adicional_eur'] = base_importacion * coste('arancel_adicional_eur_coche_proxy')
    e['prima_co2_eur'] = out['ventas'] * coste('prima_co2_eur_coche')
    e['ahorro_cambio_eur'] = base_importacion * coste('ahorro_cambio_eur_coche_proxy')
    e['cvp_despues_vm_eur'] = e['beneficio_bruto_eur'] - e['vm_eur']
    e['contribucion_marca_eur'] = (e['cvp_despues_vm_eur'] - e['coste_stock_marca_eur']
        - e['coste_transfers_eur'] - e['arancel_adicional_eur'] - e['prima_co2_eur'] + e['ahorro_cambio_eur'])
    incremento = (vector('beneficio_bruto_marca_eur') - vector('vm_real_0_90_eur')
                 - coste('coste_neto_adicional_eur_coche_proxy'))
    e['oportunidad_ventas_perdidas_eur'] = out['ventas_perdidas'] * np.maximum(0, incremento)
    e['intereses_concesionarios_eur'] = vector('pff_medio') / c['dias_anio_financiero'] * (
        c['tipo_credito_0_90'] * out['dias_coche_0_90'] + c['tipo_credito_91_180'] * out['dias_coche_91_180'])
    e['principal_vencido_nuevo_eur'] = out['vencimientos_credito_nuevos'] * vector('precio_conc_eur')
    e['ventas_csuv_vm_imputado'] = out['ventas_mas_180'] * np.array([s == 'CSUV' for s in SEGMENTOS])
    verificar_valoracion(out, e)
    return e


def verificar_valoracion(out, e):
    if any(a.shape != out['ventas'].shape or not np.isfinite(a).all() for a in e.values()):
        raise AssertionError('Importes no finitos o con dimensiones incorrectas.')
    np.testing.assert_array_equal(out['ventas'], e['ventas_stock_A012'] + e['ventas_nuevas_y_transfers'])
    np.testing.assert_allclose(e['contribucion_marca_eur'], e['beneficio_bruto_eur'] - e['vm_eur']
        - e['coste_stock_marca_eur'] - e['coste_transfers_eur'] - e['arancel_adicional_eur']
        - e['prima_co2_eur'] + e['ahorro_cambio_eur'], atol=1e-7)


def anuales_economicos(e):
    filas = []
    n = next(iter(e.values())).shape[1]
    for i, escenario in enumerate(ORDEN):
        for j, segmento in enumerate(SEGMENTOS + ['TOTAL']):
            filas.append(pd.DataFrame({'escenario': escenario, 'segmento': segmento,
                'replica': np.arange(n), **{k: a[i, :, :, j].sum(axis=1) if segmento != 'TOTAL'
                    else a[i].sum(axis=(1, 2)) for k, a in e.items()}}))
    return pd.concat(filas, ignore_index=True)


def resumen_economico(a):
    filas = []
    for (escenario, segmento), g in a.groupby(['escenario', 'segmento']):
        for k in a.columns.difference(['escenario', 'segmento', 'replica']):
            x = g[k]
            filas.append({'escenario': escenario, 'segmento': segmento, 'metrica': k,
                'media': x.mean(), 'p05': x.quantile(.05), 'p50': x.quantile(.5),
                'p95': x.quantile(.95), 'error_montecarlo_media': x.std(ddof=1)/np.sqrt(len(x)),
                'fraccion_replicas_negativas': (x < 0).mean()})
    return pd.DataFrame(filas)


def ejecutar_costes():
    out, datos04 = cargar_trayectorias()
    c = cargar_costes(); fichas = leer_csv('segmentos.csv')
    e = valorar(out, fichas, datos04['entradas_costes'], c)
    a = anuales_economicos(e)
    fisicos = pd.read_csv(RESULTADOS / '06_simulacion_mensual/resultados_anuales_replicas.csv')
    completo = fisicos.merge(a, on=['escenario', 'segmento', 'replica'], validate='one_to_one')
    tabla5 = completo.groupby(['escenario', 'segmento']).mean(numeric_only=True).drop(columns='replica').reset_index()
    tablas = {'parametros_vm': parametros_vm(fichas, c), 'resultados_economicos_replicas': a,
              'resumen_economico': resumen_economico(a), 'tabla_5': tabla5}
    salida = RESULTADOS / '07_costes_margen'; salida.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in tablas.items(): tabla.to_csv(salida / f'{nombre}.csv', index=False)
    np.savez_compressed(salida / 'costes_mensuales_A013.npz', **e, escenarios=np.array(ORDEN), segmentos=np.array(SEGMENTOS))
    m = {'configuracion': c, 'fuentes_sha256': huellas(),
         'etapa_06_sha256': sha_archivos(RESULTADOS / '06_simulacion_mensual'),
         'salidas_sha256': sha_archivos(salida, ('.csv', '.npz')),
         'definicion_margen': 'Contribución de marca tras VM, permanencia hasta 180 días, transfers y costes adicionales del escenario. No beneficio neto: sin costes fijos ni impuestos de sociedades.',
         'perdidas': 'Oportunidad positiva de una venta fresca local; informativa, nunca deducida del margen realizado.',
         'credito': 'Intereses de red separados. Vencimiento nuevo de principal es salida de caja, no gasto de la marca.',
         'arancel_cambio': 'Proxy de coste importación sobre ventas nuevas A013 y transfers; stock inicial exento. Costes de coches no vendidos permanecen sin realizar.',
         'CSUV': 'VM >180 imputado, no dato observado. Sensibilidad en 08.'}
    (salida / 'metodo.json').write_text(json.dumps(m, ensure_ascii=False, indent=2)+'\n')
    return c, tablas, e
