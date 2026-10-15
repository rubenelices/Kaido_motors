"""Demanda histórica observada y comparaciones descriptivas. No inventa demanda perdida."""
import numpy as np
import pandas as pd

EVENTOS = ['incentivo_publico', 'adelanto_compras', 'caida_tras_adelanto', 'subida_iva']

def preparar_demanda(ventas, macro):
    """Una fila original por mes, segmento y canal; suma conservada al unir eventos."""
    clave = ['t', 'segmento', 'canal']
    if ventas.duplicated(clave).any() or macro.duplicated('t').any():
        raise ValueError('La unión necesita claves únicas; hay duplicados.')
    requeridas = ['unidades', 'u_automatriculacion']
    if ventas[requeridas].isna().any().any():
        raise ValueError('No se puede calcular clientes si faltan unidades o automatriculaciones.')
    if (ventas[requeridas] < 0).any().any() or (ventas.u_automatriculacion > ventas.unidades).any():
        raise ValueError('Unidades o automatriculaciones incoherentes.')
    d = ventas.copy()
    d['clientes_observados'] = d.unidades - d.u_automatriculacion
    d = d.merge(macro[['t'] + EVENTOS], on='t', how='left', validate='many_to_one')
    if d[EVENTOS].isna().any().any():
        raise ValueError('Hay meses de ventas sin información de sucesos.')
    primera = ventas[ventas.unidades > 0].groupby('segmento').t.min()
    d['antes_de_primera_venta'] = d.t < d.segmento.map(primera)
    if len(d) != len(ventas) or d.unidades.sum() != ventas.unidades.sum():
        raise ValueError('La unión ha alterado el número de filas o de matriculaciones.')
    return d.sort_values(clave).reset_index(drop=True)

def agregar_mensual(demanda):
    """Se suma por segmento antes de comparar con un stock compartido por los canales."""
    return demanda.groupby(['t', 'anio', 'mes', 'segmento'], as_index=False).agg(
        matriculaciones=('unidades', 'sum'), automatriculaciones=('u_automatriculacion', 'sum'),
        clientes_observados=('clientes_observados', 'sum'),
        **{c: (c, 'max') for c in EVENTOS},
        renovacion_gama=('renovacion_gama', 'max'),
        antes_de_primera_venta=('antes_de_primera_venta', 'all'))

def resumir_anual(demanda):
    r = demanda.groupby('anio', as_index=False).agg(
        matriculaciones=('unidades', 'sum'), automatriculaciones=('u_automatriculacion', 'sum'),
        clientes_observados=('clientes_observados', 'sum'))
    r['peso_automatriculacion_pct'] = 100 * r.automatriculaciones / r.matriculaciones.where(r.matriculaciones > 0)
    r['crecimiento_clientes_pct'] = 100 * r.clientes_observados.pct_change(fill_method=None)
    return r

def comparar_anios(demanda, dimension=None, anterior='A011', actual='A012'):
    """El porcentaje no existe si el denominador es cero; un lanzamiento no crece infinito."""
    metricas = ['unidades', 'u_automatriculacion', 'clientes_observados']
    d = demanda[demanda.anio.isin([anterior, actual])].copy()
    if dimension is None:
        d['ambito'] = 'TOTAL'; dimension = 'ambito'
    p = d.groupby([dimension, 'anio'])[metricas].sum().unstack('anio')
    r = pd.DataFrame(index=p.index)
    alias = {'unidades': 'matriculaciones', 'u_automatriculacion': 'automatriculaciones',
             'clientes_observados': 'clientes'}
    for variable, nombre in alias.items():
        antes = p[(variable, anterior)]; despues = p[(variable, actual)]
        r[nombre + '_' + anterior] = antes
        r[nombre + '_' + actual] = despues
        r['variacion_' + nombre] = despues - antes
        r['crecimiento_' + nombre + '_pct'] = 100 * (despues / antes.where(antes > 0) - 1)
    return r.reset_index()

def comparaciones_mismo_perimetro(demanda, macro, anterior='A011', actual='A012'):
    """Mismos segmentos existentes en el año anterior y mismos meses en ambos años."""
    gama = set(demanda[(demanda.anio == anterior) & (demanda.unidades > 0)].segmento)
    d = demanda[demanda.segmento.isin(gama)]
    sucesos = macro[macro.anio.isin([anterior, actual])].copy()
    sucesos['hay_suceso'] = sucesos[EVENTOS].max(axis=1)
    comunes = sucesos.groupby('mes').agg(afectado=('hay_suceso', 'max'), anios=('anio', 'nunique'))
    meses_sin_sucesos = comunes[(comunes.afectado == 0) & (comunes.anios == 2)].index.tolist()
    ventanas = [('Año completo', list(range(1, 13))),
                ('Enero a octubre', list(range(1, 11))),
                ('Meses sin sucesos marcados en ambos años', meses_sin_sucesos)]
    filas = []
    for nombre, meses in ventanas:
        if not meses: continue
        totales = d[d.mes.isin(meses)].groupby('anio').clientes_observados.sum()
        antes = totales[anterior]; despues = totales[actual]
        filas.append({'comparacion': nombre, 'segmentos': ', '.join(sorted(gama)),
                      'meses': ', '.join(map(str, meses)), 'clientes_' + anterior: int(antes),
                      'clientes_' + actual: int(despues), 'variacion_coches': int(despues-antes),
                      'crecimiento_pct': 100 * (despues/antes - 1) if antes > 0 else np.nan})
    return pd.DataFrame(filas)

def marcar_stock(mensual, stock, umbral=0.15):
    """Indicio al cierre, no prueba de ruptura durante el mes ni estimación de ventas perdidas."""
    if umbral <= 0: raise ValueError('El umbral debe ser positivo.')
    if stock.duplicated(['t', 'segmento']).any(): raise ValueError('Stock duplicado por mes y segmento.')
    if mensual.duplicated(['t', 'segmento']).any(): raise ValueError('Agrega los canales antes de cruzar con stock.')
    d = mensual.merge(stock[['t', 'segmento', 'stock_total', 'wholesale']], on=['t', 'segmento'],
                      how='left', validate='one_to_one')
    d['stock_observado'] = d.stock_total.notna()
    d['cobertura_cierre_clientes_mes'] = d.stock_total / d.clientes_observados.where(d.clientes_observados > 0)
    d['senal_stock_bajo'] = d.stock_observado & (d.clientes_observados > 0) & (d.cobertura_cierre_clientes_mes < umbral)
    d['stock_cero_con_ventas'] = d.stock_observado & (d.clientes_observados > 0) & (d.stock_total == 0)
    return d

def referencias_estacionales(demanda, referencias=('A009', 'A010'), objetivo='A012', meses_nivel=tuple(range(1, 8)), meses_evento=(11, 12)):
    """Referencia descriptiva de igual gama, escalada por meses de nivel. No es efecto causal."""
    if len(set(meses_nivel) & set(meses_evento)):
        raise ValueError('Los meses del evento no deben calibrar el nivel de referencia.')
    # Solo segmentos con ventas antes del año objetivo: se excluye el lanzamiento CSUV.
    gama = set(demanda[(demanda.anio < objetivo) & (demanda.unidades > 0)].segmento)
    p = demanda[demanda.segmento.isin(gama)].groupby(['anio', 'mes']).clientes_observados.sum().unstack('mes')
    filas = []; curvas = []
    for referencia in referencias:
        meses = list(meses_nivel)
        denominador = p.loc[referencia, meses].sum()
        if denominador <= 0: raise ValueError('El nivel de referencia debe ser positivo.')
        escala = p.loc[objetivo, meses].sum() / denominador
        referencia_mensual = p.loc[referencia] * escala
        observado = p.loc[objetivo, list(meses_evento)].sum()
        esperado = referencia_mensual[list(meses_evento)].sum()
        filas.append({'anio_referencia': referencia, 'anio_objetivo': objetivo,
                      'meses_nivel': ', '.join(map(str, meses_nivel)),
                      'meses_evento': ', '.join(map(str, meses_evento)), 'factor_nivel': escala,
                      'clientes_observados_evento': observado,
                      'clientes_referencia_evento': esperado, 'diferencia_coches': observado-esperado})
        for mes in range(1, 13):
            curvas.append({'anio_referencia': referencia, 'mes': mes,
                           'clientes_observados': p.loc[objetivo, mes],
                           'clientes_referencia': referencia_mensual[mes]})
    return pd.DataFrame(filas), pd.DataFrame(curvas)
