"""Escenarios de demanda e incertidumbre; costes y suministro son entradas para después."""
from copy import deepcopy
import hashlib
import json
import numpy as np
import pandas as pd
from .historia import EVENTOS
from .prevision import SEGMENTOS, CANALES, huellas
from .rutas import PROYECTO, RESULTADOS, leer_csv

ORDEN = ['Pesimista', 'Central', 'Optimista']


def cargar_configuracion():
    config = json.loads((PROYECTO / 'configuracion' / 'escenarios_A013.json').read_text())
    validar_configuracion(config)
    return config


def validar_configuracion(config):
    if set(config['escenarios']) != set(ORDEN):
        raise ValueError('Se necesitan los tres escenarios, sin asignarles probabilidades.')
    if not isinstance(config['replicas'], int) or config['replicas'] < 2:
        raise ValueError('Debe haber al menos dos réplicas.')
    if not isinstance(config['bloque_meses'], int) or not 1 <= config['bloque_meses'] <= 12:
        raise ValueError('Bloque temporal entre uno y doce meses.')
    if not 0 <= config['csuv_cv_anual'] <= 1 or not 1 <= config['mes_lanzamientos'] <= 12:
        raise ValueError('Incertidumbre o mes de lanzamiento no válidos.')
    for p in config['escenarios'].values():
        if not 0 <= p['caida_meses'] <= 12 or not 0 <= p['retraso_meses'] <= 12:
            raise ValueError('Duración de sucesos no válida.')
        for campo in ['segmento_retraso', 'segmento_parada']:
            if p[campo] is not None and p[campo] not in SEGMENTOS:
                raise ValueError('Segmento de suceso desconocido.')
        for campo in ['ayuda_desde_mes', 'mes_parada_fabrica']:
            if p[campo] is not None and not 1 <= p[campo] <= 12:
                raise ValueError('Mes de suceso fuera de A013.')
        if not 0 <= p['factor_produccion_mes_parada'] <= 1 or p['factor_capacidad'] < 1:
            raise ValueError('Factores de suministro incoherentes.')
        if p['mejora_producto_pct'] < 0 or not 0 <= p['penalizacion_retraso_pct'] <= 1:
            raise ValueError('Efectos de producto incoherentes.')
        if not 0 <= p['fraccion_prima_co2'] <= 1:
            raise ValueError('Fracción CO2 fuera de [0,1].')
        if p['arancel_adicional_pct_pff'] < 0 or p['ahorro_cambio_pct_pff'] < 0:
            raise ValueError('Arancel o ahorro negativo.')


def cargar_prevision():
    carpeta = RESULTADOS / '03_prevision_validacion'
    ruta = carpeta / 'metodo.json'
    if not ruta.is_file():
        raise FileNotFoundError('Ejecuta primero 02_modelo_y_validacion.ipynb: faltan sus resultados.')
    metodo = json.loads(ruta.read_text())
    if huellas() != metodo['fuentes_sha256']:
        raise ValueError('Las fuentes han cambiado: vuelve a ejecutar los notebooks 01 y 02.')
    for nombre, digest in metodo['salidas_sha256'].items():
        if hashlib.sha256((carpeta / nombre).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Se ha modificado {nombre}. Regenera 02_modelo_y_validacion.ipynb.')
    nombres = ['prevision_base_A013', 'macro_base_A013', 'coeficientes_A013',
               'errores_fuera_muestra', 'renovaciones_historicas', 'metricas_A012']
    return {n: pd.read_csv(carpeta / f'{n}.csv') for n in nombres}


def construir_escenarios(base, macro_base, coeficientes, segmentos, config):
    """Cambia economía/sucesos en el mercado y producto en la cuota, sin tocar suministro."""
    validar_configuracion(config)
    if base.duplicated(['t', 'segmento', 'canal']).any() or len(base) != 12 * 7 * 3:
        raise ValueError('La base debe contener doce meses, siete segmentos y tres canales.')
    if sorted(base.t.unique()) != list(range(145, 157)):
        raise ValueError('La base no corresponde a A013.')
    macro_base = macro_base.sort_values('t').reset_index(drop=True)
    if macro_base.t.tolist() != list(range(145, 157)):
        raise ValueError('Macro base incompleta o duplicada.')
    calendarios, previsiones = [], []
    fichas = segmentos.set_index('segmento')
    for nombre in ORDEN:
        p = config['escenarios'][nombre]
        macro = macro_base.copy()
        macro['paro'] += p['paro_pp']
        macro['confianza_consumidor'] += p['confianza_puntos_geopolitica']
        macro['tipo_credito_consumo'] += p['credito_pp']
        macro['viajeros_var'] += p['viajeros_pp']
        macro[EVENTOS] = 0
        macro.loc[macro.mes <= p['caida_meses'], 'caida_tras_adelanto'] = 1
        if p['ayuda_desde_mes'] is not None:
            macro.loc[macro.mes >= p['ayuda_desde_mes'], 'incentivo_publico'] = 1
        macro['escenario'] = nombre
        calendarios.append(macro)
        for s in SEGMENTOS:
            fraccion_nueva = fichas.loc[s, 'codigos_nuevos_A013'] / fichas.loc[s, 'codigos_A013']
            for c in CANALES:
                z = base[(base.segmento == s) & (base.canal == c)].sort_values('t').copy()
                b = coeficientes[(coeficientes.segmento == s) & (coeficientes.canal == c)]
                log_ratio = np.zeros(12)
                for variable, beta in zip(b.variable, b.coeficiente_log):
                    log_ratio += beta * (macro[variable].to_numpy() - macro_base[variable].to_numpy())
                z['mercado_escenario'] = np.maximum(0, (z.mercado_previsto + 1) * np.exp(log_ratio) - 1)
                lanzamiento = config['mes_lanzamientos'] + (p['retraso_meses'] if s == p['segmento_retraso'] else 0)
                factor = np.ones(12)
                factor[z.mes.to_numpy() >= lanzamiento] += p['mejora_producto_pct'] * fraccion_nueva
                if s == p['segmento_retraso']:
                    retraso = (z.mes.to_numpy() >= config['mes_lanzamientos']) & (z.mes.to_numpy() < lanzamiento)
                    factor[retraso] -= p['penalizacion_retraso_pct'] * fraccion_nueva
                z['factor_producto'] = factor
                z['cuota_escenario'] = np.clip(z.cuota * factor, 0, 1)
                z['demanda_media'] = z.mercado_escenario * z.cuota_escenario
                z['escenario'] = nombre
                previsiones.append(z)
    return pd.concat(previsiones, ignore_index=True), pd.concat(calendarios, ignore_index=True)


def preparar_residuos(errores):
    """Errores mensuales por segmento, sin triplicar los canales ni usar CSUV prelaunch."""
    z = errores[errores.segmento != 'CSUV'].groupby(['t', 'segmento'])[['real', 'prevision']].sum()
    r = np.log((z.real + 0.5) / (z.prevision + 0.5)).unstack('segmento')
    if r.isna().any().any() or r.index.tolist() != list(range(109, 145)):
        raise ValueError('Se requieren errores fuera de muestra completos de A010 a A012.')
    r = r - r.mean()
    r['CSUV'] = (r['SSUV'] + r['MSUV']) / 2
    return r[SEGMENTOS]


def generar_factores(residuos, replicas=1000, semilla=42, bloque=3, csuv_cv=0.30):
    """Bootstrap circular multivariante y riesgo anual de lanzamiento para CSUV.

    Un mismo índice remuestrea todos los segmentos: conserva sus shocks comunes.
    La corrección exp asegura factores de media teórica uno, sin recortar colas.
    """
    if replicas < 2 or not 1 <= bloque <= 12 or not 0 <= csuv_cv <= 1:
        raise ValueError('Parámetros de incertidumbre no válidos.')
    e = residuos[SEGMENTOS].to_numpy(dtype=float)
    if not np.isfinite(e).all() or len(e) < 12:
        raise ValueError('Residuos no finitos o demasiado cortos.')
    rng = np.random.default_rng(semilla)
    comienzos = rng.integers(0, len(e), size=(replicas, int(np.ceil(12 / bloque))))
    indices = ((comienzos[:, :, None] + np.arange(bloque)) % len(e)).reshape(replicas, -1)[:, :12]
    factores = np.exp(e[indices]) / np.exp(e).mean(axis=0)
    sigma = np.sqrt(np.log1p(csuv_cv ** 2))
    anual = np.exp(rng.normal(-sigma ** 2 / 2, sigma, size=replicas))
    factores[:, :, SEGMENTOS.index('CSUV')] *= anual[:, None]
    return factores


def simular_demanda(previsiones, factores):
    simuladas = []
    for nombre in ORDEN:
        p = previsiones[previsiones.escenario == nombre].groupby(['mes', 'segmento']).demanda_media.sum().unstack('segmento')
        if p.index.tolist() != list(range(1, 13)) or set(p.columns) != set(SEGMENTOS):
            raise ValueError('Previsión incompleta o meses fuera de orden.')
        media = p[SEGMENTOS].to_numpy()
        simuladas.append(np.rint(media[None, :, :] * factores).astype(np.int64))
    return np.stack(simuladas)


def resumir_simulaciones(simuladas, previsiones):
    anuales, mensuales = [], []
    for i, nombre in enumerate(ORDEN):
        determinista = previsiones[previsiones.escenario == nombre].groupby('segmento').demanda_media.sum()
        for j, s in enumerate(SEGMENTOS + ['TOTAL']):
            d = simuladas[i, :, :, j] if s != 'TOTAL' else simuladas[i].sum(axis=2)
            anual = d.sum(axis=1)
            anuales.append({'escenario': nombre, 'segmento': s,
                            'prevision_media': determinista[s] if s != 'TOTAL' else determinista.sum(),
                            'media_simulada': anual.mean(), 'desviacion_anual': anual.std(ddof=1),
                            'p05': np.quantile(anual, 0.05), 'p50': np.quantile(anual, 0.50),
                            'p95': np.quantile(anual, 0.95),
                            'error_montecarlo_media': anual.std(ddof=1) / np.sqrt(len(anual))})
            for mes in range(12):
                mensuales.append({'escenario': nombre, 'segmento': s, 'mes': mes + 1,
                                  'media_simulada': d[:, mes].mean(),
                                  'p10': np.quantile(d[:, mes], 0.10), 'p50': np.quantile(d[:, mes], 0.50),
                                  'p90': np.quantile(d[:, mes], 0.90)})
    return pd.DataFrame(anuales), pd.DataFrame(mensuales)


def entradas_coste_suministro(segmentos, config):
    costes, suministro = [], []
    for nombre in ORDEN:
        p = config['escenarios'][nombre]
        for _, s in segmentos.iterrows():
            costes.append({'escenario': nombre, 'segmento': s.segmento,
                           'arancel_adicional_eur_coche_proxy': s.pff_medio * p['arancel_adicional_pct_pff'],
                           'prima_co2_eur_coche': s.prima_co2_eur * p['fraccion_prima_co2'],
                           'ahorro_cambio_eur_coche_proxy': s.pff_medio * p['ahorro_cambio_pct_pff'],
                           'coste_neto_adicional_eur_coche_proxy':
                           s.pff_medio * (p['arancel_adicional_pct_pff'] - p['ahorro_cambio_pct_pff'])
                           + s.prima_co2_eur * p['fraccion_prima_co2']})
            for mes in range(1, 13):
                parada = s.segmento == p['segmento_parada'] and mes == p['mes_parada_fabrica']
                suministro.append({'escenario': nombre, 'segmento': s.segmento, 'mes_fabricacion': mes,
                                   't_fabricacion': 144 + mes, 't_llegada': 144 + mes + 3,
                                   'factor_produccion': p['factor_produccion_mes_parada'] if parada else 1.0,
                                   'factor_capacidad': p['factor_capacidad']})
    return pd.DataFrame(costes), pd.DataFrame(suministro)


def sensibilidad(datos, segmentos, config, residuos):
    filas = []
    for duracion in [1, 3, 6]:
        alternativa = deepcopy(config)
        alternativa['escenarios']['Central']['caida_meses'] = duracion
        p, _ = construir_escenarios(datos['prevision_base_A013'], datos['macro_base_A013'],
                                    datos['coeficientes_A013'], segmentos, alternativa)
        central = p[p.escenario == 'Central'].groupby('segmento').demanda_media.sum()
        filas.append({'hipotesis': 'Duración de caída central (meses)', 'valor': duracion,
                      'demanda_total_central': central.sum(), 'csuv_media': central['CSUV'],
                      'csuv_p05': np.nan, 'csuv_p95': np.nan})
    p, _ = construir_escenarios(datos['prevision_base_A013'], datos['macro_base_A013'],
                                datos['coeficientes_A013'], segmentos, config)
    for cv in [0.15, 0.30, 0.45]:
        f = generar_factores(residuos, config['replicas'], config['semilla'], config['bloque_meses'], cv)
        anual, _ = resumir_simulaciones(simular_demanda(p, f), p)
        c = anual[(anual.escenario == 'Central') & (anual.segmento == 'CSUV')].iloc[0]
        filas.append({'hipotesis': 'CV anual adicional CSUV', 'valor': cv,
                      'demanda_total_central': p[p.escenario == 'Central'].demanda_media.sum(),
                      'csuv_media': c.prevision_media, 'csuv_p05': c.p05, 'csuv_p95': c.p95})
    return pd.DataFrame(filas)


def descomponer_efectos(datos, segmentos, config):
    """Cuenta incremental en un orden declarado; las interacciones dependen del orden."""
    grupos = {
        'Economía': ['paro_pp', 'credito_pp'],
        'Combustible (proxy confianza)': ['confianza_puntos_geopolitica'],
        'Turismo': ['viajeros_pp'],
        'Duración de caída': ['caida_meses'],
        'Plan de ayuda': ['ayuda_desde_mes'],
        'Producto': ['mejora_producto_pct', 'segmento_retraso', 'retraso_meses', 'penalizacion_retraso_pct'],
    }
    filas = []
    for nombre in ['Pesimista', 'Optimista']:
        actual = deepcopy(config)
        actual['escenarios'][nombre] = deepcopy(config['escenarios']['Central'])
        def totales():
            p, _ = construir_escenarios(datos['prevision_base_A013'], datos['macro_base_A013'],
                                        datos['coeficientes_A013'], segmentos, actual)
            return p[p.escenario == nombre].groupby('segmento').demanda_media.sum()
        anterior = totales()
        for grupo, campos in grupos.items():
            for campo in campos:
                actual['escenarios'][nombre][campo] = config['escenarios'][nombre][campo]
            nuevo = totales()
            for s in SEGMENTOS + ['TOTAL']:
                antes = anterior[s] if s != 'TOTAL' else anterior.sum()
                despues = nuevo[s] if s != 'TOTAL' else nuevo.sum()
                filas.append({'escenario': nombre, 'segmento': s, 'componente': grupo,
                              'demanda_antes': antes, 'demanda_despues': despues,
                              'cambio_coches': despues - antes})
            anterior = nuevo
    return pd.DataFrame(filas)


def ejecutar_escenarios():
    salida = RESULTADOS / '04_escenarios_incertidumbre'
    salida.mkdir(parents=True, exist_ok=True)
    config = cargar_configuracion()
    datos = cargar_prevision()
    segmentos = leer_csv('segmentos.csv')
    previsiones, macros = construir_escenarios(datos['prevision_base_A013'], datos['macro_base_A013'],
                                              datos['coeficientes_A013'], segmentos, config)
    residuos = preparar_residuos(datos['errores_fuera_muestra'])
    factores = generar_factores(residuos, config['replicas'], config['semilla'],
                               config['bloque_meses'], config['csuv_cv_anual'])
    simuladas = simular_demanda(previsiones, factores)
    anual, mensual = resumir_simulaciones(simuladas, previsiones)
    costes, suministro = entradas_coste_suministro(segmentos, config)
    tablas = {'prevision_escenarios_A013': previsiones, 'macro_escenarios_A013': macros,
              'resumen_anual': anual, 'bandas_mensuales': mensual,
              'residuos_calibracion': residuos.reset_index(),
              'correlacion_residuos': residuos.corr().rename_axis('segmento').reset_index(),
              'entradas_costes': costes, 'entradas_suministro': suministro,
              'descomposicion_efectos': descomponer_efectos(datos, segmentos, config),
              'sensibilidad_hipotesis': sensibilidad(datos, segmentos, config, residuos)}
    for nombre, tabla in tablas.items():
        tabla.to_csv(salida / f'{nombre}.csv', index=False)
    np.savez_compressed(salida / 'replicas_demanda_A013.npz', demanda=simuladas,
                        escenarios=np.array(ORDEN), segmentos=np.array(SEGMENTOS),
                        meses=np.arange(1, 13), semilla=np.array(config['semilla']))
    metodo = {'configuracion': config, 'fuentes_sha256': huellas(),
              'prevision_03_sha256': datos_sha256(),
              'intervalos': 'Condicionales al modelo, a cada escenario y a las hipótesis; sin cobertura futura garantizada',
              'escenarios': 'No se asignan probabilidades; mismos shocks por réplica en los tres escenarios',
              'calibracion': '36 meses anidados A010–A012 de seis segmentos; CSUV usa proxy SSUV/MSUV',
              'incertidumbre': 'Bloques circulares conjuntos de tres meses, multiplicadores media 1; CSUV factor anual adicional',
              'limites': 'No corrige sesgo sistemático ni identifica demanda perdida; base 03 no supera ingenua total en A012',
              'salidas': 'Demanda potencial de planificación basada en compras observadas; ventas realizables dependen del stock',
              'suministro': 'Choque de fabricación llega tres meses después; se aplicará al plan común central en 05',
              'costes': 'Arancel/cambio sobre PFF son proxies; prima CO2 viene del caso; todavía no se calcula margen',
              'csuv_cv_anual': '30% es una hipótesis de riesgo de lanzamiento, con sensibilidad 15%/45%',
              'salidas_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(salida.iterdir()) if p.suffix in ['.csv', '.npz']}}
    (salida / 'metodo.json').write_text(json.dumps(metodo, ensure_ascii=False, indent=2) + '\n')
    return config, tablas, simuladas


def datos_sha256():
    carpeta = RESULTADOS / '03_prevision_validacion'
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(carpeta.iterdir()) if p.suffix in ['.json', '.csv']}
