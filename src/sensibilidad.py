"""Etapa 08: comparación pareada de políticas, sin probabilidades de escenarios."""
from copy import deepcopy
import hashlib
import json
import numpy as np
import pandas as pd
from .rutas import RESULTADOS, leer_csv
from .costes import (cargar_costes, cargar_trayectorias, valorar, anuales_economicos)
from .produccion import cargar_politica, cargar_escenarios, preparar_plan, sha_archivos
from .simulacion import simular_trayectorias, resultados_anuales
from .prevision import huellas


VARIANTES = [
    ('Base', {}), ('Sin revisión', {'revision_activa': False}),
    ('Pedidos registrados', {'observacion_revision': 'pedidos'}),
    ('Sin transfers', {'fraccion_transfer_deficit': 0}),
    ('Transfer de todo el déficit rentable', {'fraccion_transfer_deficit': 1}),
    ('Pipeline -25%', {'pipeline_multiplicador': .75}),
    ('Pipeline +25%', {'pipeline_multiplicador': 1.25}),
    ('Seguridad mitad', {'multiplicador_seguridad': .5}),
    ('Seguridad +50%', {'multiplicador_seguridad': 1.5}),
    ('Edades iniciales jóvenes', {'edades_iniciales': 'jovenes'}),
    ('Edades iniciales mayores', {'edades_iniciales': 'mayores'}),
    ('Seguridad 75%', {'multiplicador_seguridad': .75}),
    ('Seguridad 125%', {'multiplicador_seguridad': 1.25}),
    ('Seguridad doble', {'multiplicador_seguridad': 2}),
    ('Seguridad mitad y transfer completo', {'multiplicador_seguridad': .5, 'fraccion_transfer_deficit': 1}),
    ('Seguridad +50% y transfer completo', {'multiplicador_seguridad': 1.5, 'fraccion_transfer_deficit': 1}),
]
# Los barcos y las edades no son decisiones. Los pedidos exigirían un dato nuevo.
NO_DECISIONES = {'Pipeline -25%', 'Pipeline +25%', 'Edades iniciales jóvenes',
                'Edades iniciales mayores', 'Pedidos registrados'}


def comparacion_pareada(candidato, base):
    """Diferencia por réplica, no resta de cuantiles de distribuciones distintas."""
    claves = ['escenario', 'segmento', 'replica']
    campos = ['contribucion_marca_eur', 'ventas_perdidas', 'stock_medio', 'servicio_unidades_pct']
    z = candidato[claves+campos].merge(base[claves+campos], on=claves, suffixes=('', '_base'), validate='one_to_one')
    if len(z) != len(base) or len(z) != len(candidato):
        raise ValueError('Las réplicas no corresponden entre políticas.')
    filas = []
    for (e, s), g in z.groupby(['escenario', 'segmento']):
        for k in campos:
            d = g[k] - g[k+'_base']
            filas.append({'escenario': e, 'segmento': s, 'metrica': k,
                'delta_media': d.mean(), 'delta_p05': d.quantile(.05), 'delta_p95': d.quantile(.95),
                'error_montecarlo_delta_media': d.std(ddof=1)/np.sqrt(len(d))})
    return pd.DataFrame(filas)


def ranking_robusto(resumen):
    """Máximo del peor margen medio entre escenarios, entre alternativas probadas.

    No combina historias con pesos ni demuestra un óptimo global.
    """
    z = resumen.query('segmento == "TOTAL"').copy()
    z = z[~z.variante.isin(NO_DECISIONES)]
    r = z.groupby('variante').agg(peor_margen_medio_eur=('contribucion_marca_eur', 'min'),
        peor_p05_margen_eur=('contribucion_p05_eur', 'min'),
        peor_servicio_medio_pct=('servicio_unidades_pct', 'min'),
        mayor_stock_medio=('stock_medio', 'max')).reset_index()
    r['cumple_servicio_medio_95_tres_escenarios'] = r.peor_servicio_medio_pct >= 95
    return r.sort_values('peor_margen_medio_eur', ascending=False)


def ejecutar_sensibilidad():
    out_base, datos04 = cargar_trayectorias()
    c = cargar_costes(); politica = cargar_politica(); fichas = leer_csv('segmentos.csv')
    d07 = RESULTADOS / '07_costes_margen'
    m07 = json.loads((d07 / 'metodo.json').read_text())
    if m07['configuracion'] != c or m07['fuentes_sha256'] != huellas():
        raise ValueError('Costes o fuentes cambiados: regenera el bloque de costes del notebook 04.')
    for carpeta, registro in [(d07, m07['salidas_sha256']),
            (RESULTADOS / '06_simulacion_mensual', m07['etapa_06_sha256'])]:
        for nombre, h in registro.items():
            if hashlib.sha256((carpeta / nombre).read_bytes()).hexdigest() != h:
                raise ValueError(f'Entrada alterada: {nombre}. Regenera el bloque de costes del notebook 04.')
    demanda, _, _ = cargar_escenarios()
    plan_base = preparar_plan(demanda, datos04, politica)
    def anual(out, costes=c, entradas=datos04['entradas_costes']):
        return resultados_anuales(out).merge(anuales_economicos(valorar(out, fichas, entradas, costes)),
            on=['escenario', 'segmento', 'replica'], validate='one_to_one')
    base = anual(out_base)
    res, diferencias = [], []
    for nombre, cambios in VARIANTES:
        p = deepcopy(politica); p.update(cambios)
        if nombre == 'Base':
            a = base
        else:
            plan = preparar_plan(demanda, datos04, p)
            plan['capacidad'] = plan_base['capacidad'].copy()
            out = simular_trayectorias(demanda, plan, p, datos04['entradas_suministro'], plan['politica_transfers'])
            a = anual(out)
        g = a.groupby(['escenario', 'segmento'])
        z = g.mean(numeric_only=True).drop(columns='replica').reset_index()
        q = g.contribucion_marca_eur.quantile([.05, .95]).unstack().rename(columns={.05: 'contribucion_p05_eur', .95: 'contribucion_p95_eur'}).reset_index()
        z = z.merge(q, on=['escenario', 'segmento']); z.insert(0, 'variante', nombre)
        z['tipo'] = 'Hipótesis o información adicional' if nombre in NO_DECISIONES else 'Política operativa'
        res.append(z)
        delta = comparacion_pareada(a, base); delta.insert(0, 'variante', nombre); diferencias.append(delta)
    economicas = []
    for nombre, cambios in [('VM CSUV viejo sin incremento', {'csuv_vm_mas_180_factor': 1}),
            ('VM CSUV viejo +50%', {'csuv_vm_mas_180_factor': 1.5}),
            ('Permanencia marca 1 EUR/día', {'coste_marca_eur_dia': 1}),
            ('Permanencia marca 3 EUR/día', {'coste_marca_eur_dia': 3}),
            ('Arancel y cambio en todo retail', {'base_arancel_cambio': 'todo_retail'})]:
        cfg = dict(c); cfg.update(cambios)
        a = anual(out_base, cfg)
        z = a.groupby(['escenario', 'segmento']).mean(numeric_only=True).drop(columns='replica').reset_index()
        z.insert(0, 'hipotesis', nombre); economicas.append(z)
    entradas = datos04['entradas_costes'].copy()
    entradas['prima_co2_eur_coche'] = 0
    entradas['coste_neto_adicional_eur_coche_proxy'] -= datos04['entradas_costes'].prima_co2_eur_coche
    z = anual(out_base, entradas=entradas).groupby(['escenario', 'segmento']).mean(numeric_only=True).drop(columns='replica').reset_index()
    z.insert(0, 'hipotesis', 'Matriz no carga CO2'); economicas.append(z)
    resumen = pd.concat(res, ignore_index=True)
    tablas = {'comparacion_politicas': resumen, 'diferencias_pareadas': pd.concat(diferencias, ignore_index=True),
              'ranking_robusto': ranking_robusto(resumen), 'sensibilidad_costes': pd.concat(economicas, ignore_index=True)}
    salida = RESULTADOS / '08_sensibilidad_recomendaciones'; salida.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in tablas.items(): tabla.to_csv(salida / f'{nombre}.csv', index=False)
    metodo = {'fuentes_sha256': huellas(), 'costes': c, 'politica_base': politica,
        'etapa_07_sha256': sha_archivos(d07), 'replicas_por_variante_escenario': demanda.shape[1],
        'variantes': [{'nombre': n, 'cambios': v} for n, v in VARIANTES],
        'criterio': 'Maximizar peor media de contribución entre tres escenarios dentro de políticas ensayadas. Filtro adicional: servicio medio >=95% en cada escenario.',
        'sensibilidad_costes': 'Mismas trayectorias físicas; aísla valoración, sin volver a decidir transfers ni producción.',
        'limites': 'Muestras pareadas, capacidad física común. No probabilidad de escenarios, no optimización global, selección y evaluación en mismas réplicas.',
        'salidas_sha256': sha_archivos(salida, ('.csv',))}
    (salida / 'metodo.json').write_text(json.dumps(metodo, ensure_ascii=False, indent=2)+'\n')
    return tablas
