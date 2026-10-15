"""Etapa 06: inventario FIFO, revisión sin futuro y fabricación a tres meses.

No calcula el margen anual. Registra unidades, edades y días-coche para el 07.
Los transfers llegan para su comprador y no quedan almacenados en la red.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np
import pandas as pd
from .prevision import SEGMENTOS, huellas
from .escenarios import ORDEN
from .produccion import (cargar_politica, validar_politica, cargar_escenarios, preparar_plan,
                          pedido_a_tres_meses, sha_archivos)
from .rutas import RESULTADOS


def cohortes_iniciales(stock, modo='centros'):
    """Siete cohortes: tres por tramo <=180 y una >180; cantidades enteras."""
    edades = {'centros': [195, 165, 135, 105, 75, 45, 15],
              'jovenes': [181, 151, 121, 91, 61, 31, 1],
              'mayores': [210, 180, 150, 120, 90, 60, 30]}[modo]
    z = stock.set_index('segmento').loc[SEGMENTOS]
    cantidades = np.zeros((7, 19), dtype=np.int64)
    cantidades[:, 0] = z.s_mas_180.to_numpy(dtype=np.int64)
    for campo, posiciones in [('s_91_180', [1, 2, 3]), ('s_0_90', [4, 5, 6])]:
        for j, total in enumerate(z[campo]):
            n = int(total)
            cantidades[j, posiciones] = [n // 3 + int(k < n % 3) for k in range(3)]
    if (cantidades < 0).any() or not np.array_equal(cantidades.sum(axis=1), z.stock_total):
        raise ValueError('Las cohortes iniciales no conservan el stock.')
    return cantidades, np.array(edades)


def consumir_fifo(cantidades, demanda):
    """Venta por orden de cohortes, de más antiguas a más recientes."""
    anteriores = np.cumsum(cantidades, axis=-1) - cantidades
    vendidos = np.minimum(cantidades, np.maximum(0, np.asarray(demanda)[..., None] - anteriores))
    return cantidades - vendidos, vendidos


def exposiciones_edad(vendidos, restantes, edades, dia_venta=15, dias_mes=30):
    """Tiempo de permanencia en cada tramo y nuevas expiraciones de crédito.

    Solo el tiempo de A013: no vuelve a cargar los días pasados en A012.
    La financiación termina en día 180; el vencimiento sucede al pasar de 180.
    """
    def dias_tramo(d, inicio, fin):
        return np.maximum(0, np.minimum(edades + d, fin) - np.maximum(edades, inicio))
    expos90 = (vendidos * dias_tramo(dia_venta, 0, 90) + restantes * dias_tramo(dias_mes, 0, 90)).sum(axis=-1)
    expos180 = (vendidos * dias_tramo(dia_venta, 90, 180) + restantes * dias_tramo(dias_mes, 90, 180)).sum(axis=-1)
    vence_vendido = (edades <= 180) & (edades + dia_venta > 180)
    vence_restante = (edades <= 180) & (edades + dias_mes > 180)
    vencidos = (vendidos * vence_vendido + restantes * vence_restante).sum(axis=-1)
    edad_venta = edades + dia_venta
    v90 = vendidos[..., edad_venta <= 90].sum(axis=-1)
    v180 = vendidos[..., (edad_venta > 90) & (edad_venta <= 180)].sum(axis=-1)
    vmas = vendidos[..., edad_venta > 180].sum(axis=-1)
    return {'dias_coche_0_90': expos90, 'dias_coche_91_180': expos180,
            'vencimientos_credito_nuevos': vencidos,
            'ventas_0_90': v90, 'ventas_91_180': v180, 'ventas_mas_180': vmas}


def factores_suministro(tabla):
    if tabla.duplicated(['escenario', 'mes_fabricacion', 'segmento']).any() or len(tabla) != 3*12*7:
        raise ValueError('Suministro incompleto o duplicado.')
    if not (tabla.t_llegada - tabla.t_fabricacion).eq(3).all():
        raise ValueError('El suministro debe respetar tres meses de plazo.')
    salida = {}
    for nombre in ORDEN:
        z = tabla[tabla.escenario == nombre]
        salida[nombre] = {c: z.pivot(index='mes_fabricacion', columns='segmento', values=c).reindex(
            index=range(1, 13), columns=SEGMENTOS).to_numpy() for c in ['factor_produccion', 'factor_capacidad']}
        if any(not np.isfinite(v).all() for v in salida[nombre].values()):
            raise ValueError('Faltan factores de suministro.')
    return salida


def simular_trayectorias(demanda, plan, politica, suministro, politica_transfers):
    """Consume demanda mes a mes. El controlador solo observa pasado/presente.

    La historia del escenario no modifica el plan común. Solo la fábrica aplica
    el choque del mes, y la decisión de transfer usa el margen supuesto conocido.
    """
    validar_politica(politica)
    if demanda.ndim != 4 or demanda.shape[0] != 3 or demanda.shape[2:] != (12, 7):
        raise ValueError('Demanda con forma escenarios × réplicas × 12 × 7.')
    if not np.issubdtype(demanda.dtype, np.integer) or (demanda < 0).any():
        raise ValueError('Demanda entera no negativa requerida.')
    if plan['media'].shape != (15, 7) or plan['llegadas'].shape != (15, 7):
        raise ValueError('El plan debe cubrir las tres llegadas posteriores al cierre.')
    n = demanda.shape[1]
    campos_enteros = ['demanda', 'ventas_locales', 'transfers', 'ventas', 'ventas_perdidas',
        'stock_inicio', 'llegadas', 'stock_final', 'stock_mas_90', 'stock_mas_180',
        'ventas_0_90', 'ventas_91_180', 'ventas_mas_180', 'dias_coche_0_90', 'dias_coche_91_180',
        'vencimientos_credito_nuevos', 'fabricar_solicitado', 'fabricado', 'reduccion_capacidad',
        'produccion_anulada', 'en_camino_cierre', 'observacion_revision']
    campos_reales = ['stock_medio', 'factor_revision', 'prevision_llegada_t_mas_3',
                      'stock_proyectado_t_mas_2', 'capacidad_disponible']
    out = {c: np.zeros(demanda.shape, dtype=np.int64 if c in campos_enteros else float)
           for c in campos_enteros + campos_reales}
    out['demanda'] = demanda.copy()
    condiciones = factores_suministro(suministro)
    coh_inicial, edades_iniciales = cohortes_iniciales(plan['stock_inicial'], politica['edades_iniciales'])
    inicial = coh_inicial.sum(axis=1)
    nominal = plan['capacidad'].set_index('segmento').loc[SEGMENTOS].capacidad_nominal_supuesta.to_numpy()
    for escenario, nombre in enumerate(ORDEN):
        inventario = np.broadcast_to(coh_inicial, (n, 7, 19)).copy()
        llegadas = np.broadcast_to(plan['llegadas'], (n, 15, 7)).copy()
        revision = np.ones((n, 7))
        permitidos = politica_transfers[politica_transfers.escenario == nombre].set_index('segmento').loc[SEGMENTOS].transfer_permitido.to_numpy(dtype=bool)
        for i in range(12):
            edades = np.r_[edades_iniciales + 30*i, np.maximum(0, (i-np.arange(12))*30)]
            inicio = inventario.sum(axis=2)
            inventario[:, :, 7+i] += llegadas[:, i]
            locales = np.minimum(demanda[escenario, :, i], inventario.sum(axis=2))
            inventario, vendidos = consumir_fifo(inventario, locales)
            deficit = demanda[escenario, :, i] - locales
            transfers = np.floor(deficit * politica['fraccion_transfer_deficit']).astype(np.int64) * permitidos
            perdidas = deficit - transfers
            fin = inventario.sum(axis=2)
            exposicion = exposiciones_edad(vendidos, inventario, edades, politica['dia_venta'])
            exposicion['ventas_0_90'] += transfers
            observadas = locales + transfers if politica['observacion_revision'] == 'ventas' else demanda[escenario, :, i]
            out['observacion_revision'][escenario, :, i] = observadas
            desde = max(0, i-politica['ventana_revision_meses']+1)
            denominador = plan['media'][desde:i+1].sum(axis=0)
            numerador = out['observacion_revision'][escenario, :, desde:i+1].sum(axis=1)
            ratio = np.divide(numerador, denominador, out=np.ones_like(numerador, dtype=float), where=denominador > 0)
            if politica['revision_activa']:
                peso = politica['peso_nueva_revision']
                revision = (1-peso)*revision + peso*np.clip(ratio, politica['revision_minima'], politica['revision_maxima'])
            proximas = plan['media'][None, i+1:i+3] * revision[:, None, :]
            objetivo = plan['media'][i+3] * revision
            # Transposición convierte el eje meses en el sumado por la regla del caso.
            solicitud, proyectado = pedido_a_tres_meses(fin, np.swapaxes(llegadas[:, i+1:i+3], 0, 1),
                np.swapaxes(proximas, 0, 1), objetivo, plan['ss'])
            if not politica['revision_activa']:
                solicitud = np.broadcast_to(plan['llegadas'][i+3], (n, 7)).copy()
            capacidad = np.floor(nominal * condiciones[nombre]['factor_capacidad'][i]).astype(np.int64)
            con_capacidad = np.minimum(solicitud, capacidad)
            fabricado = np.floor(con_capacidad * condiciones[nombre]['factor_produccion'][i]).astype(np.int64)
            llegadas[:, i+3] = fabricado
            fila = {'ventas_locales': locales, 'transfers': transfers, 'ventas': locales+transfers,
                    'ventas_perdidas': perdidas, 'stock_inicio': inicio, 'llegadas': llegadas[:, i],
                    'stock_final': fin, 'stock_medio': inicio+llegadas[:, i]-locales*(30-politica['dia_venta'])/30,
                    'stock_mas_90': inventario[:, :, edades+30 > 90].sum(axis=2),
                    'stock_mas_180': inventario[:, :, edades+30 > 180].sum(axis=2),
                    'fabricar_solicitado': solicitud, 'fabricado': fabricado,
                    'reduccion_capacidad': solicitud-con_capacidad,
                    'produccion_anulada': con_capacidad-fabricado,
                    'en_camino_cierre': llegadas[:, i+1:i+4].sum(axis=1),
                    'factor_revision': revision.copy(), 'prevision_llegada_t_mas_3': objetivo,
                    'stock_proyectado_t_mas_2': proyectado, 'capacidad_disponible': np.broadcast_to(capacidad, (n, 7)),
                    **exposicion}
            for c, valores in fila.items():
                out[c][escenario, :, i] = valores
    verificar_trayectorias(out, plan)
    return out


def verificar_trayectorias(out, plan):
    """Identidades exactas de demanda, stock, edades y calendario de fabricación."""
    if any(not np.isfinite(a).all() for a in out.values()):
        raise AssertionError('Una trayectoria contiene un valor no finito.')
    for c in ['demanda', 'ventas', 'ventas_locales', 'transfers', 'ventas_perdidas', 'stock_final',
              'stock_mas_90', 'stock_mas_180', 'fabricado', 'dias_coche_0_90', 'dias_coche_91_180']:
        if (out[c] < 0).any(): raise AssertionError(f'Cantidad negativa: {c}')
    if not np.array_equal(out['demanda'], out['ventas']+out['ventas_perdidas']):
        raise AssertionError('No se conserva la demanda.')
    if not np.array_equal(out['stock_inicio']+out['llegadas']-out['ventas_locales'], out['stock_final']):
        raise AssertionError('No se conserva el stock local.')
    if not np.array_equal(out['ventas'], out['ventas_0_90']+out['ventas_91_180']+out['ventas_mas_180']):
        raise AssertionError('Las ventas por edad no suman las ventas totales.')
    if not np.array_equal(out['stock_inicio'][:, :, 1:], out['stock_final'][:, :, :-1]):
        raise AssertionError('Hay una discontinuidad entre meses.')
    if not np.array_equal(out['llegadas'][:, :, 3:], out['fabricado'][:, :, :9]):
        raise AssertionError('La fabricación no llega a tres meses.')
    for i in range(3):
        if not (out['llegadas'][:, :, i] == plan['llegadas'][i]).all():
            raise AssertionError('Se ha alterado una de las tres primeras llegadas comunes.')
    if not (out['stock_mas_180'] <= out['stock_mas_90']).all() or not (out['stock_mas_90'] <= out['stock_final']).all():
        raise AssertionError('Tramos de antigüedad incompatibles.')
    if not np.array_equal(out['fabricar_solicitado'], out['fabricado']+out['reduccion_capacidad']+out['produccion_anulada']):
        raise AssertionError('Pedidos y restricciones no concilian.')


SUMAS = ['demanda', 'ventas', 'ventas_locales', 'transfers', 'ventas_perdidas', 'llegadas',
         'fabricar_solicitado', 'fabricado', 'reduccion_capacidad', 'produccion_anulada',
         'ventas_0_90', 'ventas_91_180', 'ventas_mas_180', 'dias_coche_0_90', 'dias_coche_91_180',
         'vencimientos_credito_nuevos']


def resultados_anuales(out):
    """Total por réplica antes de cuantiles. Medias y coberturas no se suman."""
    n = out['demanda'].shape[1]
    filas = []
    for e, nombre in enumerate(ORDEN):
        for j, s in enumerate(SEGMENTOS + ['TOTAL']):
            serie = lambda c: out[c][e, :, :, j] if s != 'TOTAL' else out[c][e].sum(axis=2)
            valores = {c: serie(c).sum(axis=1) for c in SUMAS}
            valores['stock_medio'] = serie('stock_medio').mean(axis=1)
            for c in ['stock_final', 'stock_mas_90', 'stock_mas_180', 'en_camino_cierre']:
                valores[c] = serie(c)[:, -1]
            valores['meses_con_ventas_perdidas'] = (serie('ventas_perdidas') > 0).sum(axis=1)
            valores['servicio_unidades_pct'] = np.divide(100*valores['ventas'], valores['demanda'],
                out=np.full(n, np.nan), where=valores['demanda'] > 0)
            valores['stock_medio_meses_demanda'] = np.divide(12*valores['stock_medio'], valores['demanda'],
                out=np.full(n, np.nan), where=valores['demanda'] > 0)
            valores['stock_final_meses_demanda'] = np.divide(12*valores['stock_final'], valores['demanda'],
                out=np.full(n, np.nan), where=valores['demanda'] > 0)
            valores['stock_mas_90_final_pct'] = np.divide(100*valores['stock_mas_90'], valores['stock_final'],
                out=np.full(n, np.nan), where=valores['stock_final'] > 0)
            bloque = pd.DataFrame(valores)
            bloque.insert(0, 'replica', np.arange(n))
            bloque.insert(0, 'segmento', s); bloque.insert(0, 'escenario', nombre)
            filas.append(bloque)
    return pd.concat(filas, ignore_index=True)


def resumir_anuales(anuales):
    filas = []
    campos = [c for c in anuales.columns if c not in ['escenario', 'segmento', 'replica']]
    for (e, s), z in anuales.groupby(['escenario', 'segmento']):
        for c in campos:
            x = z[c].dropna()
            filas.append({'escenario': e, 'segmento': s, 'metrica': c, 'media': x.mean(),
                          'p05': x.quantile(.05) if len(x) else np.nan,
                          'p50': x.quantile(.5) if len(x) else np.nan,
                          'p95': x.quantile(.95) if len(x) else np.nan,
                          'error_montecarlo_media': x.std()/np.sqrt(len(x)) if len(x) > 1 else np.nan,
                          'replicas_con_valor': len(x)})
    return pd.DataFrame(filas)


def resumir_mensuales(out):
    campos = ['demanda', 'ventas', 'ventas_perdidas', 'transfers', 'llegadas', 'stock_final',
              'stock_mas_90', 'stock_mas_180', 'fabricado', 'factor_revision']
    filas = []
    for e, nombre in enumerate(ORDEN):
        for j, s in enumerate(SEGMENTOS + ['TOTAL']):
            for i in range(12):
                fila = {'escenario': nombre, 'segmento': s, 'mes': i+1}
                for c in campos:
                    if c == 'factor_revision' and s == 'TOTAL': continue  # Ratios por segmento no se suman.
                    x = out[c][e, :, i, j] if s != 'TOTAL' else out[c][e, :, i].sum(axis=1)
                    fila[c+'_media'] = x.mean()
                    if c in ['stock_final', 'ventas_perdidas']:
                        fila[c+'_p05'] = np.quantile(x, .05)
                        fila[c+'_p95'] = np.quantile(x, .95)
                filas.append(fila)
    return pd.DataFrame(filas)


def ejemplos_mensuales(out, numero=5):
    n = min(numero, out['demanda'].shape[1])
    etiquetas = {'escenario': np.repeat(ORDEN, n*12*7),
                  'replica': np.tile(np.repeat(np.arange(n), 12*7), 3),
                  'mes': np.tile(np.repeat(np.arange(1, 13), 7), 3*n),
                  'segmento': np.tile(SEGMENTOS, 3*n*12)}
    etiquetas.update({c: a[:, :n].reshape(-1) for c, a in out.items()})
    return pd.DataFrame(etiquetas)


def sensibilidad_simulacion(demanda, datos04, politica):
    """Mismas 1.000 réplicas en cada variante; no se asignan probabilidades a variantes."""
    variantes = [('Base', {}), ('Sin revisión', {'revision_activa': False}),
                 ('Pedidos registrados', {'observacion_revision': 'pedidos'}),
                 ('Sin transfers', {'fraccion_transfer_deficit': 0.0}),
                 ('Transfer de todo el déficit rentable', {'fraccion_transfer_deficit': 1.0}),
                 ('Pipeline -25%', {'pipeline_multiplicador': .75}),
                 ('Pipeline +25%', {'pipeline_multiplicador': 1.25}),
                 ('Seguridad mitad', {'multiplicador_seguridad': .5}),
                 ('Seguridad +50%', {'multiplicador_seguridad': 1.5}),
                 ('Edades iniciales jóvenes', {'edades_iniciales': 'jovenes'}),
                 ('Edades iniciales mayores', {'edades_iniciales': 'mayores'})]
    filas = []
    capacidad_base = preparar_plan(demanda, datos04, politica)['capacidad'].copy()
    for nombre, cambios in variantes:
        p = deepcopy(politica); p.update(cambios)
        plan = preparar_plan(demanda, datos04, p)
        # La fábrica física es la misma al cambiar barcos, seguridad o información.
        plan['capacidad'] = capacidad_base.copy()
        out = simular_trayectorias(demanda, plan, p, datos04['entradas_suministro'], plan['politica_transfers'])
        a = resultados_anuales(out)
        resumen = a.groupby(['escenario', 'segmento']).mean(numeric_only=True).drop(columns='replica').reset_index()
        resumen.insert(0, 'variante', nombre)
        filas.append(resumen)
    return pd.concat(filas, ignore_index=True)


def ejecutar_simulacion():
    politica = cargar_politica()
    demanda, datos04, _ = cargar_escenarios()
    carpeta05 = RESULTADOS / '05_produccion_stock'
    if not (carpeta05 / 'metodo.json').is_file():
        raise FileNotFoundError('Ejecuta primero el bloque de producción del notebook 03.')
    m05 = json.loads((carpeta05 / 'metodo.json').read_text())
    if m05['politica'] != politica:
        raise ValueError('La política ha cambiado: regenera producción antes de la simulación en el notebook 03.')
    for carpeta, registro in [(carpeta05, m05['salidas_sha256']),
            (RESULTADOS / '04_escenarios_incertidumbre', m05['etapa_04_sha256']),
            (RESULTADOS / '03_prevision_validacion', m05['etapa_03_sha256'])]:
        for nombre, h in registro.items():
            if hashlib.sha256((carpeta / nombre).read_bytes()).hexdigest() != h:
                raise ValueError(f'Entrada alterada: {nombre}. Regenera el bloque de producción del notebook 03.')
    plan = preparar_plan(demanda, datos04, politica)
    with np.load(carpeta05 / 'plan_central.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(f['llegadas_plan'], plan['llegadas'])
        np.testing.assert_allclose(f['demanda_media_horizonte'], plan['media'])
    out = simular_trayectorias(demanda, plan, politica, datos04['entradas_suministro'], plan['politica_transfers'])
    anual = resultados_anuales(out)
    tablas = {'resultados_anuales_replicas': anual, 'resumen_anual': resumir_anuales(anual),
              'resumen_mensual': resumir_mensuales(out), 'ejemplos_mensuales_5_replicas': ejemplos_mensuales(out),
              'sensibilidad_operativa': sensibilidad_simulacion(demanda, datos04, politica)}
    salida = RESULTADOS / '06_simulacion_mensual'
    salida.mkdir(parents=True, exist_ok=True)
    for n, t in tablas.items(): t.to_csv(salida / f'{n}.csv', index=False)
    np.savez_compressed(salida / 'trayectorias_A013.npz', **out, escenarios=np.array(ORDEN),
                        segmentos=np.array(SEGMENTOS), meses=np.arange(1, 13))
    metodo = {'politica': politica, 'fuentes_sha256': huellas(), 'etapa_05_sha256': sha_archivos(carpeta05),
              'etapa_04_sha256': sha_archivos(RESULTADOS / '04_escenarios_incertidumbre'),
              'replicas_por_escenario': demanda.shape[1], 'trayectorias_mes_segmento': int(np.prod(demanda.shape)),
              'revision': 'Ventas realizadas hasta el mes actual; pedidos solo en sensibilidad con observabilidad supuesta',
              'FIFO': 'Cohortes iniciales y nuevas; venta en día 15, llegadas inicio de mes, 30 días por mes',
              'transfer': 'Cobertura 50% del déficit rentable, sin permanencia en red; no resta stock local',
              'datos_para_07': 'Ventas por edad, días-coche de crédito/stock y vencimientos nuevos; no margen anual calculado',
              'fabricacion': 'Mes i llega i+3; primer trimestre común; trimestre final fabricado llega A014',
              'salidas_sha256': sha_archivos(salida, ('.csv', '.npz'))}
    (salida / 'metodo.json').write_text(json.dumps(metodo, ensure_ascii=False, indent=2) + '\n')
    return politica, tablas, out
