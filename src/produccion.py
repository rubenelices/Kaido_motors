"""Etapa 05: hipótesis de logística, stock de seguridad y plan central común.

Mes 0 = cierre de diciembre A012. Se decide L3 en mes 0; fabricar en enero
A013 implica llegada en abril. El plan incluye la continuación ene–mar A014.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from .prevision import SEGMENTOS, huellas
from .escenarios import ORDEN, cargar_prevision, cargar_configuracion
from .rutas import PROYECTO, RESULTADOS, leer_csv


def cargar_politica():
    p = json.loads((PROYECTO / 'configuracion/politica_stock.json').read_text())
    validar_politica(p)
    return p


def validar_politica(p):
    if not .5 < p['nivel_servicio_objetivo'] < 1:
        raise ValueError('El servicio objetivo debe estar entre 0,5 y 1, sin incluirlos.')
    if p['metodo_seguridad'] not in ['max_formula_y_bootstrap', 'formula', 'bootstrap']:
        raise ValueError('Método de seguridad desconocido.')
    for c in ['multiplicador_seguridad', 'pipeline_multiplicador', 'continuacion_A014_multiplicador']:
        if not np.isfinite(p[c]) or p[c] < 0:
            raise ValueError(f'{c} debe ser finito y no negativo.')
    if not np.isfinite(p['capacidad_multiplicador_plan']) or p['capacidad_multiplicador_plan'] < 1:
        raise ValueError('La capacidad nominal debe cubrir el máximo pedido del plan común.')
    if not 0 <= p['fraccion_transfer_deficit'] <= 1 or not 0 <= p['peso_nueva_revision'] <= 1:
        raise ValueError('Fracción transfer o peso de revisión fuera de [0,1].')
    if p['observacion_revision'] not in ['ventas', 'pedidos']:
        raise ValueError('La revisión usa ventas o pedidos registrados.')
    if not isinstance(p['ventana_revision_meses'], int) or not 1 <= p['ventana_revision_meses'] <= 12:
        raise ValueError('Ventana de revisión entre uno y doce meses.')
    if not 0 < p['revision_minima'] <= 1 <= p['revision_maxima']:
        raise ValueError('Límites de revisión incoherentes.')
    if p['dias_mes'] != 30 or not isinstance(p['dia_venta'], int) or not 0 < p['dia_venta'] <= 30:
        raise ValueError('La discretización utiliza meses de 30 días y venta dentro del mes.')
    for c in ['revision_activa', 'permitir_transfers_no_rentables']:
        if not isinstance(p[c], bool):
            raise ValueError(f'{c} debe ser un booleano.')
    if p['edades_iniciales'] not in ['centros', 'jovenes', 'mayores']:
        raise ValueError('Distribución de edades iniciales desconocida.')


def sha_archivos(carpeta, sufijos=('.csv', '.json', '.npz')):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(carpeta.iterdir()) if p.suffix in sufijos}


def cargar_escenarios():
    """Rechaza fuentes, configuración o artefactos 03/04 modificados sin regenerar."""
    cargar_prevision()
    carpeta = RESULTADOS / '04_escenarios_incertidumbre'
    ruta = carpeta / 'metodo.json'
    if not ruta.is_file():
        raise FileNotFoundError('Ejecuta primero el bloque de escenarios del notebook 03.')
    m = json.loads(ruta.read_text())
    if m['fuentes_sha256'] != huellas() or m['configuracion'] != cargar_configuracion():
        raise ValueError('Fuentes o hipótesis de escenarios modificadas: regenera los notebooks 02 y 03.')
    for nombre, h in m['prevision_03_sha256'].items():
        p = RESULTADOS / '03_prevision_validacion' / nombre
        if hashlib.sha256(p.read_bytes()).hexdigest() != h:
            raise ValueError('La previsión cambió después de los escenarios. Regenera el notebook 03.')
    for nombre, h in m['salidas_sha256'].items():
        if hashlib.sha256((carpeta / nombre).read_bytes()).hexdigest() != h:
            raise ValueError(f'Escenario alterado: {nombre}. Regenera el notebook 03.')
    with np.load(carpeta / 'replicas_demanda_A013.npz', allow_pickle=False) as f:
        demanda = f['demanda'].copy()
        if f['escenarios'].tolist() != ORDEN or f['segmentos'].tolist() != SEGMENTOS:
            raise ValueError('Orden de escenarios o segmentos no coincide.')
        if f['meses'].tolist() != list(range(1, 13)):
            raise ValueError('Los meses de las réplicas no son A013 completo.')
    if demanda.shape != (3, m['configuracion']['replicas'], 12, 7):
        raise ValueError('Dimensiones de las réplicas no válidas.')
    if not np.issubdtype(demanda.dtype, np.integer) or demanda.min() < 0:
        raise ValueError('La demanda debe estar en coches enteros no negativos.')
    tablas = {n: pd.read_csv(carpeta / f'{n}.csv') for n in
              ['prevision_escenarios_A013', 'macro_escenarios_A013', 'entradas_costes', 'entradas_suministro']}
    return demanda, tablas, m


def extender_central(prevision_base, coeficientes, multiplicador=1.0, caida_por_mes=None):
    """Tres meses supuestos de A014, sin caída, con calendario Q1 A013 repetido.

    No usa observaciones A014. Mantiene macro y cuota y prolonga la tendencia
    temporal 12 meses desde Q1 A013. Solo sirve a los pedidos del final de A013.
    """
    d = prevision_base.copy()
    d['mes_plan'] = d.t - 144
    d['continuacion_supuesta'] = False
    extension = d[d.mes.isin([1, 2, 3])].copy()
    cuotas_cierre = d[d.mes == 12].set_index(['segmento', 'canal']).cuota
    for indice, fila in extension.iterrows():
        c = coeficientes[(coeficientes.segmento == fila.segmento) & (coeficientes.canal == fila.canal)].set_index('variable').coeficiente_log
        caida = 1 if caida_por_mes is None else caida_por_mes.loc[fila.mes]
        cambio = 12 * c['t'] - caida * c['caida_tras_adelanto']
        mercado = max(0, (fila.mercado_previsto + 1) * np.exp(cambio) - 1)
        extension.loc[indice, 'mercado_previsto'] = mercado * multiplicador
        cuota = cuotas_cierre.loc[(fila.segmento, fila.canal)]
        extension.loc[indice, 'cuota'] = cuota
        extension.loc[indice, 'prevision'] = mercado * cuota * multiplicador
    extension['t'] += 12
    extension['mes_plan'] += 12
    extension['continuacion_supuesta'] = True
    return pd.concat([d, extension], ignore_index=True).sort_values(['mes_plan', 'segmento', 'canal'])


def seguridad(central_media, central_replicas, politica):
    """Colchón constante por segmento: fórmula y errores acumulados de tres meses."""
    if central_media.shape != (12, 7) or central_replicas.shape[1:] != (12, 7):
        raise ValueError('Media o réplicas centrales no tienen doce meses y siete segmentos.')
    nivel = politica['nivel_servicio_objetivo']
    z = float(norm.ppf(nivel))
    sigma = np.sqrt(np.mean(np.var(central_replicas, axis=0, ddof=1), axis=0))
    formula = z * sigma * np.sqrt(3)
    errores = central_replicas - central_media[None, :, :]
    acumulados = np.concatenate([errores[:, i:i+3].sum(axis=1) for i in range(10)], axis=0)
    empirico = np.maximum(0, np.quantile(acumulados, nivel, axis=0))
    metodos = {'formula': formula, 'bootstrap': empirico,
               'max_formula_y_bootstrap': np.maximum(formula, empirico)}
    elegido = np.ceil(metodos[politica['metodo_seguridad']] * politica['multiplicador_seguridad']).astype(np.int64)
    return pd.DataFrame({'segmento': SEGMENTOS, 'sigma_mensual_coches': sigma, 'nivel_objetivo': nivel,
                         'z': z, 'formula_independiente': formula, 'cuantil_error_tres_meses': empirico,
                         'stock_seguridad': elegido, 'metodo': politica['metodo_seguridad']})


def pipeline_inicial(stock, multiplicador=1.0):
    """Proxy de llegadas enero/febrero: media wholesale octubre–diciembre A012."""
    base = stock[stock.t.between(142, 144)]
    if base.duplicated(['t', 'segmento']).any() or len(base) != 21 or base.wholesale.isna().any():
        raise ValueError('Se necesitan tres meses de wholesale de los siete segmentos.')
    media = base.groupby('segmento').wholesale.mean().reindex(SEGMENTOS)
    cantidades = np.rint(media.to_numpy() * multiplicador).astype(np.int64)
    filas = [{'mes_llegada': mes, 't_llegada': 144 + mes, 'segmento': s,
              'wholesale_medio_Q4': float(media[s]), 'llegadas_comprometidas': int(cantidades[j]),
              'origen': 'Hipótesis: wholesale no es un manifiesto de barcos'}
             for mes in [1, 2] for j, s in enumerate(SEGMENTOS)]
    return pd.DataFrame(filas)


def pedido_a_tres_meses(stock_cierre, llegadas_proximas, demandas_proximas, demanda_objetivo, ss):
    """Regla literal del caso; proyección negativa es déficit previsto, no stock real."""
    proyectado = np.asarray(stock_cierre) + np.sum(llegadas_proximas, axis=0) - np.sum(demandas_proximas, axis=0)
    pedido = np.ceil(np.maximum(0, np.asarray(demanda_objetivo) + np.asarray(ss) - proyectado)).astype(np.int64)
    return pedido, proyectado


def plan_central(media_horizonte, pipeline, stock_inicial, ss):
    """Plan ex ante completo. L3 se decide en mes 0; producción A013 es L4–L15."""
    media = np.asarray(media_horizonte, dtype=float)
    if media.shape != (15, 7) or np.asarray(pipeline).shape != (2, 7):
        raise ValueError('Se necesitan quince meses de previsión y dos de pipeline.')
    if not np.isfinite(media).all() or (media < 0).any() or (np.asarray(pipeline) < 0).any():
        raise ValueError('Previsión o pipeline inválidos.')
    llegadas = np.zeros((15, 7), dtype=np.int64)
    llegadas[:2] = pipeline
    inicial = np.asarray(stock_inicial, dtype=float)
    llegadas[2], proyectado = pedido_a_tres_meses(inicial, llegadas[:2], media[:2], media[2], ss)
    calculo_marzo = pd.DataFrame({'segmento': SEGMENTOS, 'stock_diciembre_A012': inicial,
                                 'llegadas_enero_febrero': llegadas[:2].sum(axis=0),
                                 'demanda_enero_febrero': media[:2].sum(axis=0),
                                 'stock_proyectado_febrero': proyectado, 'demanda_marzo': media[2],
                                 'stock_seguridad': ss, 'pedido_decidido_diciembre_llega_marzo': llegadas[2]})
    stock = inicial.copy()
    registros = []
    for i in range(12):
        disponible = stock + llegadas[i]
        ventas = np.minimum(disponible, media[i])
        fin = disponible - ventas
        pedido, proyectado = pedido_a_tres_meses(fin, llegadas[i+1:i+3], media[i+1:i+3], media[i+3], ss)
        llegadas[i+3] = pedido
        for j, s in enumerate(SEGMENTOS):
            registros.append({'mes': i+1, 'segmento': s, 'stock_inicio': stock[j], 'llegadas': llegadas[i, j],
                              'demanda_central': media[i, j], 'ventas_locales_esperadas': ventas[j],
                              'deficit_esperado_sin_transfer': media[i, j] - ventas[j], 'stock_cierre_esperado': fin[j],
                              'stock_proyectado_t_mas_2': proyectado[j], 'fabricar_plan_inicial': pedido[j],
                              't_fabricacion': 145+i, 't_llegada': 148+i,
                              'anio_llegada': 'A013' if i+4 <= 12 else 'A014',
                              'mes_llegada': (i+3) % 12+1, 'stock_seguridad': ss[j]})
        stock = fin
    return llegadas, pd.DataFrame(registros), calculo_marzo


def preparar_plan(demanda, datos04, politica):
    validar_politica(politica)
    datos03 = cargar_prevision()
    # El central vigente del 04 puede haberse editado: no usar una base 03 desactualizada.
    base = datos04['prevision_escenarios_A013'].query('escenario == "Central"').copy()
    base['prevision'] = base.demanda_media
    base['mercado_previsto'] = base.mercado_escenario
    base['cuota'] = base.cuota_escenario
    base = base[['t', 'mes', 'segmento', 'canal', 'prevision', 'mercado_previsto',
                 'cuota', 'meses_cuota', 'regla_cuota']]
    caida = datos04['macro_escenarios_A013'].query('escenario == "Central"').set_index('mes').caida_tras_adelanto
    horizonte = extender_central(base, datos03['coeficientes_A013'],
                                 politica['continuacion_A014_multiplicador'], caida)
    media = horizonte.groupby(['mes_plan', 'segmento']).prevision.sum().unstack('segmento')[SEGMENTOS].to_numpy()
    tabla_ss = seguridad(media[:12], demanda[ORDEN.index('Central')], politica)
    ss = tabla_ss.stock_seguridad.to_numpy()
    stock = leer_csv('stock_mensual.csv')
    inicial = stock[stock.t == 144].set_index('segmento').loc[SEGMENTOS].reset_index()
    if len(inicial) != 7 or not np.array_equal(inicial.stock_total,
                                            inicial[['s_0_90', 's_91_180', 's_mas_180']].sum(axis=1)):
        raise ValueError('Stock inicial incompleto o incoherente.')
    pip = pipeline_inicial(stock, politica['pipeline_multiplicador'])
    matriz_pip = pip.pivot(index='mes_llegada', columns='segmento', values='llegadas_comprometidas')[SEGMENTOS].to_numpy()
    llegadas, plan, calculo = plan_central(media, matriz_pip, inicial.stock_total.to_numpy(), ss)
    capacidad = np.ceil(llegadas[3:].max(axis=0) * politica['capacidad_multiplicador_plan']).astype(np.int64)
    plan['capacidad_nominal_supuesta'] = plan.segmento.map(dict(zip(SEGMENTOS, capacidad)))
    # Política de transfers: solo el margen incremental conocido, sin cálculo de resultado anual.
    fichas = leer_csv('segmentos.csv').set_index('segmento')
    costes = datos04['entradas_costes'].copy()
    costes['beneficio_bruto_base'] = costes.segmento.map(fichas.beneficio_bruto_marca_eur)
    costes['vm_transfer_fresco'] = costes.segmento.map(fichas.vm_real_0_90_eur)
    costes['coste_transfer_unitario'] = .20 * costes.beneficio_bruto_base
    costes['margen_incremental_transfer_proxy'] = (costes.beneficio_bruto_base - costes.vm_transfer_fresco
        - costes.coste_neto_adicional_eur_coche_proxy - costes.coste_transfer_unitario)
    costes['transfer_permitido'] = politica['permitir_transfers_no_rentables'] | (costes.margen_incremental_transfer_proxy > 0)
    tabla_cap = pd.DataFrame({'segmento': SEGMENTOS, 'max_pedido_plan_A013': llegadas[3:].max(axis=0),
                              'capacidad_nominal_supuesta': capacidad})
    return {'horizonte': horizonte, 'seguridad': tabla_ss, 'stock_inicial': inicial, 'pipeline_inicial': pip,
            'plan_central': plan, 'calculo_marzo': calculo, 'capacidad': tabla_cap,
            'politica_transfers': costes, 'media': media, 'llegadas': llegadas, 'ss': ss}


def sensibilidad_plan(demanda, datos04, politica):
    filas = []
    for nombre, campo, valores in [('Pipeline', 'pipeline_multiplicador', [.75, 1., 1.25]),
                                  ('Seguridad', 'multiplicador_seguridad', [.5, 1., 1.5]),
                                  ('Continuación A014', 'continuacion_A014_multiplicador', [.75, 1., 1.25])]:
        for valor in valores:
            alternativa = dict(politica); alternativa[campo] = valor
            p = preparar_plan(demanda, datos04, alternativa)
            for j, s in enumerate(SEGMENTOS + ['TOTAL']):
                elegir = lambda x: x[:, j].sum() if s != 'TOTAL' else x.sum()
                filas.append({'hipotesis': nombre, 'multiplicador': valor, 'segmento': s,
                              'llegadas_enero_febrero': elegir(p['llegadas'][:2]),
                              'llegada_marzo': elegir(p['llegadas'][2:3]),
                              'fabricar_A013_plan_inicial': elegir(p['llegadas'][3:]),
                              'llegadas_A014_comprometidas': elegir(p['llegadas'][12:])})
    return pd.DataFrame(filas)


def ejecutar_produccion():
    demanda, datos04, _ = cargar_escenarios()
    politica = cargar_politica()
    plan = preparar_plan(demanda, datos04, politica)
    salida = RESULTADOS / '05_produccion_stock'
    salida.mkdir(parents=True, exist_ok=True)
    tablas = {n: v for n, v in plan.items() if isinstance(v, pd.DataFrame)}
    tablas['sensibilidad_plan'] = sensibilidad_plan(demanda, datos04, politica)
    for n, t in tablas.items():
        t.to_csv(salida / f'{n}.csv', index=False)
    np.savez_compressed(salida / 'plan_central.npz', demanda_media_horizonte=plan['media'],
                        llegadas_plan=plan['llegadas'], stock_seguridad=plan['ss'],
                        capacidad=plan['capacidad'].capacidad_nominal_supuesta.to_numpy(),
                        stock_inicial=plan['stock_inicial'].stock_total.to_numpy(), segmentos=np.array(SEGMENTOS))
    metodo = {'politica': politica, 'fuentes_sha256': huellas(),
              'etapa_04_sha256': sha_archivos(RESULTADOS / '04_escenarios_incertidumbre'),
              'etapa_03_sha256': sha_archivos(RESULTADOS / '03_prevision_validacion'),
              'tiempo': 'Mes 0 diciembre A012 decide marzo; fabricación enero A013 llega abril; L1–L3 comunes congeladas',
              'pipeline': 'Media wholesale octubre–diciembre A012; dos meses, no dato de barcos',
              'continuacion': 'Q1 A014 supuesto: Q1 central vigente 04 sin caída, tendencia +12, mismo calendario/macro y cuota de cierre A013',
              'seguridad': 'Máximo de z*sigma*sqrt(3) y cuantil acumulado conjunto; ventanas solapadas, servicio no garantizado',
              'capacidad': f"Proxy: {100*politica['capacidad_multiplicador_plan']:g}% del mayor pedido mensual del plan inicial A013; no capacidad observada",
              'limites': 'Previsión 03 provisional; política no optimizada económicamente; edades en 06 y costes completos en 07',
              'salidas_sha256': sha_archivos(salida, ('.csv', '.npz'))}
    (salida / 'metodo.json').write_text(json.dumps(metodo, ensure_ascii=False, indent=2) + '\n')
    return politica, tablas, plan
