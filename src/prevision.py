"""Previsión de mercado y cuota con cortes temporales explícitos.

La selección no consulta A012. Los parámetros económicos tienen signos supuestos,
no interpretaciones causales. El objetivo es clientes observados, no ventas perdidas.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from .historia import EVENTOS, preparar_demanda
from .rutas import DATOS, RESULTADOS, leer_csv

SEGMENTOS = ['B', 'C', 'MD', 'SSUV', 'CSUV', 'MSUV', 'S']
CANALES = ['PARTICULARES', 'EMPRESAS_RENTING', 'ALQUILADORES']
ECONOMIA = {
    'PARTICULARES': ['paro', 'confianza_consumidor', 'tipo_credito_consumo'],
    'EMPRESAS_RENTING': ['paro', 'pib_var', 'indice_actividad'],
    'ALQUILADORES': ['paro', 'viajeros_var'],
}
NEGATIVAS = {'paro', 'tipo_credito_consumo', 'semana_santa', 'caida_tras_adelanto'}
POSITIVAS = {'confianza_consumidor', 'pib_var', 'indice_actividad', 'viajeros_var',
             'dias_laborables', 'incentivo_publico', 'adelanto_compras', 'subida_iva'}
CORTES_CV = [84, 96, 108, 120]  # A008–A011 son futuros de sus respectivos cortes.
ALPHAS = [1.0, 10.0, 30.0]
VENTANAS = [12, 24]


class RidgeConSignos(RegressorMixin, BaseEstimator):
    """Mínimos cuadrados penalizados con cotas de signo e intercepto libre."""
    def __init__(self, alpha=10.0, signos=()):
        self.alpha = alpha
        self.signos = signos

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.alpha <= 0 or len(self.signos) != X.shape[1]:
            raise ValueError('Penalización o vector de signos no válidos.')
        self.n_features_in_ = X.shape[1]
        a = np.column_stack([np.ones(len(X)), X])
        penalizacion = np.diag([0.0] + [np.sqrt(self.alpha)] * X.shape[1])
        limites_inf = np.array([-np.inf] + [0.0 if s == 1 else -np.inf for s in self.signos])
        limites_sup = np.array([np.inf] + [0.0 if s == -1 else np.inf for s in self.signos])
        solucion = lsq_linear(np.vstack([a, penalizacion]), np.r_[y, np.zeros(a.shape[1])],
                              bounds=(limites_inf, limites_sup), tol=1e-10, max_iter=1000)
        if not solucion.success:
            raise RuntimeError(f'El ajuste no ha convergido: {solucion.message}')
        self.intercept_ = float(solucion.x[0])
        self.coef_ = solucion.x[1:]
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X) @ self.coef_


def huellas():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATOS.glob('*.csv'))}


def cargar_historia():
    macro = leer_csv('macro_mensual.csv').sort_values('t').reset_index(drop=True)
    ventas = preparar_demanda(leer_csv('ventas_mensual.csv'), macro)
    mercado = leer_csv('mercado_mensual.csv').sort_values(['t', 'segmento', 'canal'])
    return ventas, mercado, macro


def proyectar_macro(macro, corte, calendario, meses_caida=None):
    """Economía Q4 persistente; turismo media anual. Solo toma pasado <= corte.

    El calendario puede proceder del histórico, pero solo usa t/mes/días/SS.
    No se leen economía ni sucesos posteriores al corte.
    """
    pasado = macro[macro.t <= corte].sort_values('t')
    if len(pasado) < 12 or int(pasado.t.max()) != corte:
        raise ValueError('El corte no tiene doce meses de historia completa.')
    cal = calendario[['t', 'mes', 'dias_laborables', 'semana_santa']].sort_values('t').copy()
    if cal.t.tolist() != list(range(corte + 1, corte + 13)):
        raise ValueError('Se necesita exactamente el calendario de los doce meses siguientes.')
    if cal.mes.tolist() != [((t - 1) % 12) + 1 for t in cal.t]:
        raise ValueError('Meses y contador temporal no coinciden.')
    columnas = [c for c in macro.select_dtypes(include='number').columns
                if c not in ['t', 'mes', 'dias_laborables', 'semana_santa']]
    for c in columnas:
        cal[c] = float(pasado.tail(3)[c].mean())
    cal['viajeros_var'] = float(pasado.tail(12).viajeros_var.mean())
    cal[EVENTOS] = 0
    if meses_caida is None:
        meses_caida = 3 if pasado.iloc[-1].adelanto_compras == 1 else 0
    if not 0 <= meses_caida <= 12:
        raise ValueError('La caída debe durar entre cero y doce meses.')
    cal.loc[cal.index[:meses_caida], 'caida_tras_adelanto'] = 1
    cal['anio'] = [f'A{(int(t)-1)//12+1:03d}' for t in cal.t]
    return cal.reset_index(drop=True)


def variables(canal):
    sucesos = [c for c in EVENTOS if c != 'incentivo_publico' or canal == 'PARTICULARES']
    return ECONOMIA[canal] + ['t', 'dias_laborables', 'semana_santa'] + sucesos


@dataclass
class ModeloMercado:
    pipeline: Pipeline
    correccion_log: float
    canal: str
    corte: int

    def predecir(self, macro):
        log = self.pipeline.predict(macro)
        return np.maximum(0, np.exp(log) * self.correccion_log - 1)

    def coeficientes(self):
        numericas = variables(self.canal)
        escala = self.pipeline.named_steps['preparar'].named_transformers_['numericas'].scale_
        beta = self.pipeline.named_steps['ajustar'].coef_
        return dict(zip(numericas, beta[:len(numericas)] / escala))


def ajustar_mercado(mercado, macro, segmento, canal, corte, alpha):
    y = mercado[(mercado.segmento == segmento) & (mercado.canal == canal) & (mercado.t <= corte)]
    x = macro[macro.t <= corte]
    d = y[['t', 'unidades']].merge(x, on='t', validate='one_to_one').sort_values('t')
    if d.t.tolist() != list(range(1, corte + 1)):
        raise ValueError('Faltan meses del mercado o de la economía antes del corte.')
    num = variables(canal)
    signos = tuple(-1 if c in NEGATIVAS else 1 if c in POSITIVAS else 0 for c in num) + (0,) * 11
    preparar = ColumnTransformer([
        ('numericas', StandardScaler(), num),
        ('meses', OneHotEncoder(categories=[list(range(1, 13))], drop='first',
                               handle_unknown='error', sparse_output=False), ['mes']),
    ])
    pipe = Pipeline([('preparar', preparar), ('ajustar', RidgeConSignos(alpha, signos))])
    pipe.fit(d, np.log1p(d.unidades))
    residuos = np.log1p(d.unidades) - pipe.predict(d)
    # Deshacer el logaritmo sin esta corrección estima una mediana, no una media.
    return ModeloMercado(pipe, float(np.exp(residuos).mean()), canal, corte)


def estimar_cuota(ventas, mercado, macro, segmento, canal, corte, ventana):
    v = ventas[(ventas.segmento == segmento) & (ventas.canal == canal) & (ventas.t <= corte)]
    positivos = v[v.clientes_observados > 0]
    if positivos.empty:
        return 0.0, 0, 'Sin historia comercial en el corte'
    inicio = max(corte - ventana + 1, int(positivos.t.min()))
    meses = macro[macro.t.between(inicio, corte)]
    normales = meses[(meses[['adelanto_compras', 'caida_tras_adelanto', 'subida_iva']] == 0).all(axis=1)].t
    # CSUV dispone de tres meses antes del adelanto de Nov–Dic; no se extrapola un crecimiento.
    usar = normales if len(normales) >= 3 else meses.t
    modo = 'Meses sin adelanto/caída/IVA' if len(normales) >= 3 else 'Todos: menos de tres meses sin sucesos'
    numerador = v[v.t.isin(usar)].clientes_observados.sum()
    denominador = mercado[(mercado.segmento == segmento) & (mercado.canal == canal) & mercado.t.isin(usar)].unidades.sum()
    if denominador <= 0 or numerador > denominador:
        raise ValueError('Cuota incoherente con el mercado.')
    return float(numerador / denominador), len(usar), modo


def historia_validable(ventas, segmento, corte):
    """Mismo perímetro comercial en todas las configuraciones de un corte."""
    positivos = ventas[(ventas.segmento == segmento) & (ventas.t <= corte) &
                       (ventas.clientes_observados > 0)]
    return not positivos.empty and int(positivos.t.min()) <= corte - 11


def predecir_corte(ventas, mercado, macro, corte, calendario, seleccion):
    futuro = proyectar_macro(macro, corte, calendario)
    filas = []
    for s in SEGMENTOS:
        config = seleccion.set_index('segmento').loc[s]
        for c in CANALES:
            mod = ajustar_mercado(mercado, macro, s, c, corte, float(config.alpha))
            q, n, modo = estimar_cuota(ventas, mercado, macro, s, c, corte, int(config.ventana))
            for t, mes, mercado_prev in zip(futuro.t, futuro.mes, mod.predecir(futuro)):
                observado = ventas[(ventas.segmento == s) & (ventas.canal == c) & (ventas.t == t)]
                naive = ventas[(ventas.segmento == s) & (ventas.canal == c) & (ventas.t == t - 12)]
                real = float(observado.clientes_observados.sum()) if len(observado) else np.nan
                filas.append({'corte': corte, 't': int(t), 'mes': int(mes), 'segmento': s, 'canal': c,
                              'real': real, 'prevision': mercado_prev * q,
                              'ingenua_estacional': float(naive.clientes_observados.sum()),
                              'mercado_previsto': mercado_prev, 'cuota': q, 'meses_cuota': n,
                              'regla_cuota': modo, 'validable': historia_validable(ventas, s, corte)})
    return pd.DataFrame(filas), futuro


def evaluar_candidatos(ventas, mercado, macro):
    """Cuatro futuros históricos; ningún objetivo supera t=132."""
    filas = []
    for corte in CORTES_CV:
        cal = macro[macro.t.between(corte + 1, corte + 12)]
        futuro = proyectar_macro(macro, corte, cal)
        for s in SEGMENTOS:
            for c in CANALES:
                reales = ventas[(ventas.segmento == s) & (ventas.canal == c) & ventas.t.between(corte + 1, corte + 12)].sort_values('t')
                naive = ventas[(ventas.segmento == s) & (ventas.canal == c) & ventas.t.between(corte - 11, corte)].sort_values('t')
                for alpha in ALPHAS:
                    p = ajustar_mercado(mercado, macro, s, c, corte, alpha).predecir(futuro)
                    for ventana in VENTANAS:
                        q, n, _ = estimar_cuota(ventas, mercado, macro, s, c, corte, ventana)
                        for i, t in enumerate(futuro.t):
                            filas.append({'corte': corte, 't': int(t), 'segmento': s, 'canal': c,
                                          'alpha': alpha, 'ventana': ventana, 'real': float(reales.iloc[i].clientes_observados),
                                          'prevision': p[i] * q, 'ingenua_estacional': float(naive.iloc[i].clientes_observados),
                                          'validable': historia_validable(ventas, s, corte)})
    return pd.DataFrame(filas)


def seleccionar(candidatos, hasta=132):
    d = candidatos[candidatos.t <= hasta]
    if d.empty:
        raise ValueError('No hay años de validación anteriores al corte.')
    filas = []
    for s in SEGMENTOS:
        z = d[(d.segmento == s) & d.validable]
        if z.empty:
            filas.append({'segmento': s, 'alpha': 10.0, 'ventana': 12, 'wape_cv_pct': np.nan,
                          'wape_ingenua_cv_pct': np.nan, 'ultimo_t_seleccion': int(d.t.max()),
                          'regla': 'Predeterminada: segmento sin historia validable'})
            continue
        puntuaciones = []
        for (a, v), grupo in z.groupby(['alpha', 'ventana']):
            p = grupo.groupby('t')[['real', 'prevision', 'ingenua_estacional']].sum()
            puntuaciones.append((float(abs(p.real - p.prevision).sum() / p.real.sum()), a, v,
                                 float(abs(p.real - p.ingenua_estacional).sum() / p.real.sum())))
        # El desempate favorece mayor regularización y ventana larga, sin consultar A012.
        error, a, v, ingenua = min(puntuaciones, key=lambda x: (x[0], -x[1], -x[2]))
        filas.append({'segmento': s, 'alpha': a, 'ventana': v, 'wape_cv_pct': 100 * error,
                      'wape_ingenua_cv_pct': 100 * ingenua, 'ultimo_t_seleccion': int(d.t.max()),
                      'regla': 'Menor WAPE mensual agregado por segmento; solo pasado'})
    return pd.DataFrame(filas)


def metricas(predicciones):
    filas = []
    # Nunca se añade CSUV al TOTAL comparable: en t=132 su lanzamiento no tiene historia.
    for s in [x for x in SEGMENTOS if x != 'CSUV'] + ['TOTAL_COMPARABLE']:
        z = predicciones[predicciones.segmento != 'CSUV'] if s == 'TOTAL_COMPARABLE' else predicciones[predicciones.segmento == s]
        p = z.groupby('t')[['real', 'prevision', 'ingenua_estacional']].sum()
        for metodo in ['prevision', 'ingenua_estacional']:
            real, previsto = p.real.to_numpy(), p[metodo].to_numpy()
            filas.append({'segmento': s, 'metodo': metodo,
                          'real_anual': real.sum(), 'previsto_anual': previsto.sum(),
                          'wape_pct': 100 * np.abs(real - previsto).sum() / real.sum(),
                          'mae_coches_mes': mean_absolute_error(real, previsto),
                          'r2': r2_score(real, previsto),
                          'sesgo_pct': 100 * (previsto - real).sum() / real.sum()})
    return pd.DataFrame(filas)


def calibrar_renovaciones(ventas, mercado, macro, corte=144):
    v = ventas[ventas.t <= corte].groupby(['t', 'segmento'], as_index=False).agg(
        clientes=('clientes_observados', 'sum'), renovacion=('renovacion_gama', 'max'))
    m = mercado[mercado.t <= corte].groupby(['t', 'segmento'], as_index=False).unidades.sum()
    z = v.merge(m, validate='one_to_one').merge(macro[['t'] + EVENTOS], validate='many_to_one')
    filas = []
    for s, g in z.groupby('segmento'):
        for t in g[g.renovacion == 1].t:
            antes = g[g.t.between(t - 3, t - 1)]
            despues = g[g.t.between(t, t + 2)]
            if len(antes) != 3 or len(despues) != 3 or antes.clientes.sum() < 60:
                continue
            q0 = antes.clientes.sum() / antes.unidades.sum()
            q1 = despues.clientes.sum() / despues.unidades.sum()
            sucesos = bool(pd.concat([antes, despues])[EVENTOS].to_numpy().any())
            filas.append({'segmento': s, 't': int(t), 'variacion_cuota_pct': 100 * (q1 / q0 - 1),
                          'hay_sucesos_en_ventana': sucesos})
    return pd.DataFrame(filas)


def ejecutar_prevision():
    salida = RESULTADOS / '03_prevision_validacion'
    salida.mkdir(parents=True, exist_ok=True)
    ventas, mercado, macro = cargar_historia()
    candidatos = evaluar_candidatos(ventas, mercado, macro)
    seleccion = seleccionar(candidatos, hasta=132)
    seleccion.to_csv(salida / 'seleccion_modelos.csv', index=False)
    # La selección queda fijada antes de hacer la única prueba final de A012.
    validacion, macro_test = predecir_corte(ventas, mercado, macro, 132,
                                          macro[macro.t.between(133, 144)], seleccion)
    errores = metricas(validacion)
    # Errores realmente posteriores a su selección: A010, A011 y la prueba A012.
    anidados = []
    elecciones = []
    for corte in [108, 120]:
        anterior = seleccionar(candidatos, hasta=corte)
        p, _ = predecir_corte(ventas, mercado, macro, corte,
                              macro[macro.t.between(corte + 1, corte + 12)], anterior)
        anidados.append(p)
        anterior['corte_prevision'] = corte
        elecciones.append(anterior)
    anidados.append(validacion)
    ultima = seleccion.copy(); ultima['corte_prevision'] = 132; elecciones.append(ultima)
    residuos = pd.concat(anidados, ignore_index=True)
    residuos['error_log'] = np.log((residuos.real + 0.5) / (residuos.prevision + 0.5))
    # Después de evaluar, se vuelve a ajustar con toda la información disponible de A012.
    futuro, macro_futuro = predecir_corte(ventas, mercado, macro, 144,
                                        leer_csv('calendario_A013.csv'), seleccion)
    # A013 no tiene ventas reales: no escribir ceros fingiendo observaciones.
    futuro = futuro.drop(columns=['real', 'ingenua_estacional', 'validable'])
    coef = []
    for s in SEGMENTOS:
        for c in CANALES:
            a = float(seleccion.set_index('segmento').loc[s, 'alpha'])
            mod = ajustar_mercado(mercado, macro, s, c, 144, a)
            coef.extend({'segmento': s, 'canal': c, 'variable': v, 'coeficiente_log': b}
                        for v, b in mod.coeficientes().items())
    tablas = {'candidatos_cv': candidatos, 'validacion_A012': validacion,
              'metricas_A012': errores, 'macro_exante_A012': macro_test,
              'errores_fuera_muestra': residuos, 'selecciones_temporales': pd.concat(elecciones, ignore_index=True),
              'prevision_base_A013': futuro, 'macro_base_A013': macro_futuro,
              'coeficientes_A013': pd.DataFrame(coef),
              'renovaciones_historicas': calibrar_renovaciones(ventas, mercado, macro)}
    for nombre, tabla in tablas.items():
        tabla.to_csv(salida / f'{nombre}.csv', index=False)
    reglas = {'fuentes_sha256': huellas(), 'corte_seleccion': 132, 'corte_prueba_final': 132,
              'corte_reajuste': 144, 'cortes_cv': CORTES_CV, 'alphas': ALPHAS, 'ventanas_cuota': VENTANAS,
              'economia': ECONOMIA, 'macro_futura': 'Media Q4 persistente; turistas media de últimos 12 meses',
              'caida_A013_meses': 3, 'objetivo': 'Clientes observados; no demanda potencial identificada',
              'CSUV': 'No validable en A012; cuota Aug-Oct A012; efecto anual de lanzamiento desconocido',
              'error_calibracion': 'Predicciones anidadas A010–A011 + prueba A012, sin selección con su año futuro',
              'signos': 'Restricciones económicas supuestas; coeficientes asociativos, no causales',
              'replica': 'Determinista; no particiones aleatorias'}
    reglas['salidas_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(salida.glob('*.csv'))}
    (salida / 'metodo.json').write_text(json.dumps(reglas, ensure_ascii=False, indent=2) + '\n')
    return seleccion, tablas
