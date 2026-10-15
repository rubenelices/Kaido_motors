"""Perfilado y controles de los datos del caso. No transforma ni corrige las fuentes."""
import itertools
import pandas as pd
import numpy as np

SEGMENTOS = ['B', 'C', 'MD', 'SSUV', 'CSUV', 'MSUV', 'S']
CANALES = ['PARTICULARES', 'EMPRESAS_RENTING', 'ALQUILADORES']
CLAVES = {
    'ventas_mensual': ['t', 'segmento', 'canal'],
    'mercado_mensual': ['t', 'segmento', 'canal'],
    'stock_mensual': ['t', 'segmento'],
    'macro_mensual': ['t'], 'calendario_A013': ['t'],
    'segmentos': ['segmento'], 'codigos': ['segmento', 'codigo_modelo', 'generacion'],
}
PERIODOS = {'ventas_mensual': (25, 144), 'mercado_mensual': (1, 144),
            'stock_mensual': (97, 144), 'macro_mensual': (1, 144),
            'calendario_A013': (145, 156)}

# Descripciones basadas en el anexo 5 y la tabla 2 del enunciado.
DESCRIPCIONES = {
    't': ('Índice mensual consecutivo; A001 enero es t=1.', 'mes ordinal'),
    'periodo': ('Etiqueta del mes del caso; no es una fecha de un año real.', 'código Axxxmm01'),
    'anio': ('Año del caso.', 'código Axxx'), 'mes': ('Mes del año.', '1 a 12'),
    'segmento': ('Grupo de coches al que pertenece el registro.', 'categoría'),
    'descripcion': ('Nombre del segmento.', 'texto'), 'canal': ('Tipo de comprador.', 'categoría'),
    'unidades': ('Matriculaciones; en códigos, ventas acumuladas del código.', 'coches'),
    'u_automatriculacion': ('De las unidades, coches matriculados por marca o concesionario.', 'coches'),
    'u_venta_especial': ('De las unidades, operaciones especiales; no son un conjunto disjunto de otros indicadores.', 'coches'),
    'u_0_90': ('Matriculaciones de coches vendidos con hasta 90 días en la red, desde A009 febrero.', 'coches'),
    'u_91_180': ('Matriculaciones con antigüedad entre 91 y 180 días.', 'coches'),
    'u_mas_180': ('Matriculaciones con antigüedad superior a 180 días.', 'coches'),
    'pvp_medio': ('Precio medio de tarifa al cliente, con impuestos.', 'euros por coche'),
    'pff_medio': ('Precio medio franco fábrica.', 'euros por coche'),
    'vm_medio': ('Gasto comercial real medio de las matriculaciones de la celda.', 'euros por coche'),
    'dias_0_90_medio': ('Días medios de los coches matriculados del primer tramo.', 'días'),
    'dias_91_180_medio': ('Días medios de los coches matriculados del segundo tramo.', 'días'),
    'dias_mas_180_medio': ('Días medios de los coches matriculados del tercer tramo.', 'días'),
    'renovacion_gama': ('Mes en que empieza a venderse un código nuevo con volumen.', '0 o 1'),
    'meses_desde_renovacion': ('Tiempo desde la última renovación de gama.', 'meses'),
    'stock_total': ('Existencias de la red al cierre del mes.', 'coches'),
    's_0_90': ('Del stock final, coches con hasta 90 días desde la factura.', 'coches'),
    's_91_180': ('Del stock final, coches entre 91 y 180 días.', 'coches'),
    's_mas_180': ('Del stock final, coches con más de 180 días.', 'coches'),
    's_sin_cliente': ('Del stock final, coches sin comprador con contrato.', 'coches'),
    'wholesale': ('Coches facturados a la red en el mes, desde A009 febrero.', 'coches'),
    'marca_propia': ('De las matriculaciones del mercado, las de Kaido.', 'coches'),
    'pct_diesel': ('Peso del diésel en el mercado de la celda.', 'fracción 0 a 1'),
    'pct_electrificado': ('Peso de híbridos, enchufables y eléctricos.', 'fracción 0 a 1'),
    'pct_renting': ('Peso del renting, disponible desde A004.', 'fracción 0 a 1'),
    'cv_medio': ('Potencia media de los coches de la celda de mercado.', 'CV'),
    'n_marcas': ('Número de marcas presentes en la celda de mercado.', 'marcas'),
    'letra': ('Letra de la familia de códigos; D se comparte entre tres segmentos.', 'texto'),
    'primer_mes': ('Primer mes de ventas; vacío para códigos futuros sin historia.', 'código Axxxmm01'),
    'ultimo_mes': ('Último mes de ventas; vacío para códigos futuros sin historia.', 'código Axxxmm01'),
    'unidades_totales': ('Matriculaciones históricas del segmento.', 'coches'),
    'unidades_ult_anio': ('Matriculaciones del segmento en A012.', 'coches'),
    'codigos': ('Número de nombres de código distintos con ventas; puede haber generaciones reutilizadas.', 'códigos'),
    'codigos_A013': ('Número de códigos en la tarifa de A013.', 'códigos'),
    'codigos_nuevos_A013': ('Códigos de A013 que no tienen historia de ventas.', 'códigos'),
    'iedmt_pct': ('Tipo del impuesto de matriculación.', 'fracción 0 a 1'),
    'margen_conc_base_pct': ('Margen base contractual del concesionario.', 'fracción 0 a 1'),
    'margen_conc_var_pct': ('Margen variable contractual del concesionario.', 'fracción 0 a 1'),
    'margen_bruto_conc_eur': ('Margen bruto del concesionario; no su beneficio neto.', 'euros por coche'),
    'precio_conc_eur': ('PFF menos margen base facturado al concesionario.', 'euros por coche'),
    'beneficio_bruto_marca_eur': ('Beneficio bruto de tarifa A013 antes de VM y prima CO2.', 'euros por coche'),
    'prima_co2_eur': ('Cargo por coche si la matriz traslada la penalización a la filial.', 'euros por coche'),
    'vm_tarifa_eur': ('VM presupuestado en la tarifa A013.', 'euros por coche'),
    'cvp_tarifa_eur': ('Beneficio bruto menos VM presupuestado.', 'euros por coche'),
    'vm_real_eur': ('VM real medio de A011 y A012.', 'euros por coche'),
    'cvp_real_eur': ('Beneficio bruto menos VM real.', 'euros por coche'),
    'vm_real_0_90_eur': ('VM real de coches vendidos en el primer tramo de edad.', 'euros por coche'),
    'vm_real_91_180_eur': ('VM real de coches vendidos en el segundo tramo.', 'euros por coche'),
    'vm_real_mas_180_eur': ('VM real de coches vendidos con más de 180 días.', 'euros por coche'),
    'credito_dia_0_90_eur': ('Interés diario del concesionario en el primer tramo.', 'euros por coche y día'),
    'credito_dia_91_180_eur': ('Interés diario del concesionario en el segundo tramo.', 'euros por coche y día'),
    'codigo_modelo': ('Nombre del código; NA es un código válido.', 'texto'),
    'generacion': ('Distingue un código reutilizado años después.', 'entero'),
    'en_tarifa_A013': ('Indica si el código está en la gama del futuro A013.', '0 o 1'),
    'pib_var': ('Variación anual del PIB.', 'porcentaje, puntos'),
    'consumo_privado_var': ('Variación anual del consumo de hogares.', 'porcentaje, puntos'),
    'paro': ('Tasa de paro.', 'porcentaje, puntos'),
    'confianza_consumidor': ('Índice de confianza de consumidores.', 'índice'),
    'hipotecas': ('Hipotecas firmadas en el mes.', 'número'),
    'tipo_credito_consumo': ('Tipo de interés del crédito al consumo.', 'porcentaje, puntos'),
    'indice_actividad': ('Índice de actividad de empresas.', 'índice'),
    'indice_comercio': ('Índice de actividad del comercio.', 'índice'),
    'viajeros_var': ('Variación anual de viajeros alojados.', 'porcentaje, puntos'),
    'incentivo_publico': ('Mes con plan público de ayuda.', '0 o 1'),
    'adelanto_compras': ('Mes con compras adelantadas por normativa.', '0 o 1'),
    'caida_tras_adelanto': ('Mes con caída después de un adelanto.', '0 o 1'),
    'subida_iva': ('Mes anterior a una subida del IVA.', '0 o 1'),
    'dias_laborables': ('Días laborables observados o del calendario facilitado.', 'días'),
    'semana_santa': ('Indica el mes con Semana Santa.', '0 o 1'),
}

def catalogar(tablas):
    """Tamaño y ausencias; no toma una ausencia como un error por defecto."""
    return pd.DataFrame([{'archivo': n, 'filas': len(d), 'columnas': len(d.columns),
        'celdas_ausentes': int(d.isna().sum().sum()), 'duplicados_completos': int(d.duplicated().sum())}
        for n, d in tablas.items()])

def perfilar(tablas):
    """Una fila por columna de cada fuente; describe el tipo observado y su distribución."""
    filas = []
    for nombre, tabla in tablas.items():
        for columna in tabla:
            s = tabla[columna]; v = s.dropna()
            descripcion, unidad = DESCRIPCIONES[columna]
            f = {'archivo': nombre, 'columna': columna, 'tipo_observado': str(s.dtype),
                 'unidad': unidad, 'descripcion': descripcion, 'ausentes': int(s.isna().sum()),
                 'pct_ausentes': 100 * s.isna().mean(), 'valores_distintos': int(v.nunique()),
                 'pct_cardinalidad': 100 * v.nunique() / len(tabla)}
            if pd.api.types.is_numeric_dtype(s):
                f.update(minimo=v.min(), maximo=v.max(), media=v.mean(), mediana=v.median(),
                         desviacion=v.std(), ceros=int((v == 0).sum()), negativos=int((v < 0).sum()))
            else:
                longitudes = v.astype(str).str.len()
                f.update(longitud_minima=longitudes.min(), longitud_maxima=longitudes.max(),
                         frecuentes='; '.join(f'{k}: {q}' for k, q in v.value_counts().head(5).items()))
            filas.append(f)
    return pd.DataFrame(filas)

def auditar(tablas):
    """Reglas del enunciado; OK es comprobado, REVISAR exige interpretación."""
    filas = []
    def regla(nombre, fallo, detalle):
        filas.append({'control': nombre, 'estado': 'OK' if int(fallo) == 0 else 'REVISAR',
                      'incidencias': int(fallo), 'detalle': detalle})
    for nombre, clave in CLAVES.items():
        d = tablas[nombre]
        regla(f'Clave completa {nombre}', d[clave].isna().any(axis=1).sum(), ', '.join(clave))
        regla(f'Clave única {nombre}', d.duplicated(clave).sum(), ', '.join(clave))
    for nombre, (inicio, fin) in PERIODOS.items():
        d = tablas[nombre]
        dimensiones = [range(inicio, fin + 1)]
        columnas = ['t']
        if 'segmento' in d: dimensiones.append(SEGMENTOS); columnas.append('segmento')
        if 'canal' in d: dimensiones.append(CANALES); columnas.append('canal')
        esperado = set(itertools.product(*dimensiones))
        observado = set(d[columnas].itertuples(index=False, name=None))
        regla(f'Cobertura completa {nombre}', len(esperado ^ observado),
              'Todas las combinaciones esperadas, incluidas las celdas de cero ventas.')
        etiqueta = d.t.map(lambda t: f'A{(int(t)-1)//12+1:03d}{(int(t)-1)%12+1:02d}01')
        fallos = (d.periodo != etiqueta) | (d.anio != etiqueta.str[:4]) | (d.mes != (d.t-1)%12+1)
        regla(f'Etiquetas coherentes {nombre}', fallos.sum(), 't, año, mes y periodo coinciden.')
    for nombre, d in tablas.items():
        for c in d:
            if c.startswith('pct_') or c.endswith('_pct'):
                v = d[c].dropna()
                regla(f'Fracción {nombre}.{c}', ((v < 0) | (v > 1)).sum(), '0,09 significa 9 %, no 0,09 %.')
            if c in ['en_tarifa_A013', 'renovacion_gama', 'semana_santa', 'incentivo_publico',
                     'adelanto_compras', 'caida_tras_adelanto', 'subida_iva']:
                regla(f'Indicador {nombre}.{c}', (~d[c].isin([0, 1])).sum(), 'Solo 0 y 1, sin ausencias.')
            if DESCRIPCIONES[c][1] == 'euros por coche' and not c.startswith('cvp_'):
                valor = d[c].dropna()
                regla(f'Importe {nombre}.{c}', ((valor < 0) | (~np.isfinite(valor))).sum(),
                      'Precio, VM o cargo no negativo. CVP puede ser negativo y se trata aparte.')
            unidad = DESCRIPCIONES[c][1]
            if unidad in ['coches', 'códigos', 'marcas', 'días', 'meses', 'entero', 'mes ordinal', '1 a 12']:
                v = d[c].dropna()
                # Las medias de días pueden ser fraccionarias; los recuentos deben ser enteros.
                mal = (v < 0) | (~np.isfinite(v))
                if unidad != 'días': mal |= v != np.floor(v)
                regla(f'Dominio {nombre}.{c}', mal.sum(), 'Cantidad no negativa; recuentos enteros.')
        if 'segmento' in d:
            regla(f'Segmentos conocidos {nombre}', (~d.segmento.isin(SEGMENTOS)).sum(), 'Siete segmentos del caso.')
        if 'canal' in d:
            regla(f'Canales conocidos {nombre}', (~d.canal.isin(CANALES)).sum(), 'Tres canales del caso.')
    v = tablas['ventas_mensual']; m = tablas['mercado_mensual']; s = tablas['stock_mensual']
    regla('Automatriculaciones contenidas', (v.u_automatriculacion > v.unidades).sum(), 'Se restan de unidades, no se suman.')
    regla('Ventas especiales contenidas', (v.u_venta_especial > v.unidades).sum(), 'No se restan: pueden ser compras de clientes.')
    regla('Kaido dentro del mercado', (m.marca_propia > m.unidades).sum(), 'Kaido está incluido en el mercado, no se añade otra vez.')
    regla('Tramos de stock', (s[['s_0_90', 's_91_180', 's_mas_180']].sum(axis=1) != s.stock_total).sum(), 'Tres tramos disjuntos.')
    regla('Stock sin cliente contenido', (s.s_sin_cliente > s.stock_total).sum(), 'Es un subconjunto, no un cuarto tramo de edad.')
    tramos = v[v.t >= 98]
    regla('Tramos de ventas desde A009 febrero', (tramos[['u_0_90', 'u_91_180', 'u_mas_180']].sum(axis=1, min_count=3) != tramos.unidades).sum(), 'Tramos de matriculaciones; no sabemos el tramo de cada automatriculación.')
    union = v[['t', 'segmento', 'canal', 'unidades']].merge(
        m[['t', 'segmento', 'canal', 'marca_propia']], on=['t', 'segmento', 'canal'], how='left', validate='one_to_one')
    regla('Mercado y ventas internas', (union.unidades != union.marca_propia).sum(), 'Comparación de todas las celdas desde A003.')
    ficha = tablas['segmentos'].set_index('segmento'); codigos = tablas['codigos']
    resumen = v.groupby('segmento').unidades.sum().reindex(ficha.index)
    regla('Ficha y ventas históricas', (resumen != ficha.unidades_totales).sum(), 'Totales acumulados por segmento.')
    ultimo = v[v.anio == 'A012'].groupby('segmento').unidades.sum().reindex(ficha.index)
    regla('Ficha y ventas A012', (ultimo != ficha.unidades_ult_anio).sum(), 'Último año por segmento.')
    por_codigo = codigos.groupby('segmento').unidades.sum().reindex(ficha.index)
    regla('Códigos y ventas históricas', (por_codigo != resumen).sum(), 'Suma de todas las generaciones.')
    gama = codigos.groupby('segmento').en_tarifa_A013.sum().reindex(ficha.index)
    nuevos = codigos[(codigos.en_tarifa_A013 == 1) & (codigos.unidades == 0)].groupby('segmento').size().reindex(ficha.index, fill_value=0)
    nombres = codigos[codigos.unidades > 0].groupby('segmento').codigo_modelo.nunique().reindex(ficha.index)
    regla('Códigos distintos con ventas', (nombres != ficha.codigos).sum(), 'Un nombre reutilizado se cuenta una sola vez en esta columna.')
    regla('Gama A013', (gama != ficha.codigos_A013).sum(), 'Códigos en tarifa por segmento.')
    regla('Códigos nuevos A013', (nuevos != ficha.codigos_nuevos_A013).sum(), 'Códigos en tarifa sin historia de ventas.')
    letras = codigos.merge(ficha[['letra']], on='segmento', suffixes=('', '_ficha'), validate='many_to_one')
    regla('Letras de códigos', ((letras.letra != letras.letra_ficha) | (letras.codigo_modelo.str[0] != letras.letra)).sum(), 'La letra D no identifica por sí sola un segmento.')
    regla('Código NA conservado', int('NA' not in set(codigos.codigo_modelo)), 'Es texto válido, nunca una ausencia.')
    for vm, cvp in [('vm_tarifa_eur', 'cvp_tarifa_eur'), ('vm_real_eur', 'cvp_real_eur')]:
        error = (ficha.beneficio_bruto_marca_eur - ficha[vm] - ficha[cvp]).abs()
        regla(f'Beneficio bruto = {vm} + {cvp}', (error > 2).sum(), 'Tolerancia declarada de 2 € por redondeo de cifras enteras.')
    return pd.DataFrame(filas)

def interpretar_ausencias(tablas):
    """Compara cada hueco con la disponibilidad y la definición del anexo 5."""
    filas = []
    for nombre, d in tablas.items():
        for c in d:
            ausente = d[c].isna()
            if not ausente.any(): continue
            motivo = 'Ausencia pendiente de interpretación.'
            esperado = pd.Series(False, index=d.index)
            decision = 'No imputar automáticamente.'
            if nombre == 'ventas_mensual':
                if c in ['pvp_medio', 'pff_medio', 'vm_medio']:
                    esperado = d.unidades == 0; motivo = 'Media indefinida en una celda sin matriculaciones.'
                elif c in ['u_0_90', 'u_91_180', 'u_mas_180']:
                    esperado = d.t < 98; motivo = 'Antigüedad de ventas disponible desde A009 febrero.'
                elif c.startswith('dias_'):
                    tramo = c.replace('dias_', 'u_').replace('_medio', '')
                    esperado = (d.t < 98) | (d[tramo] == 0)
                    motivo = 'No hay medición histórica o no hay coches en ese tramo.'
                elif c == 'meses_desde_renovacion':
                    # Aún no existe una renovación observada para ese segmento.
                    primera = d[d.renovacion_gama == 1].groupby('segmento').t.min()
                    esperado = d.t < d.segmento.map(primera).fillna(np.inf)
                    motivo = 'Aún no ha ocurrido una renovación registrada del segmento.'
            elif nombre == 'mercado_mensual':
                esperado = d.unidades == 0
                motivo = 'Composición o potencia media indefinida en una celda sin mercado.'
                if c == 'pct_renting':
                    esperado |= d.t < 37
                    motivo = 'Renting disponible desde A004; sin media si el mercado es cero.'
                if (ausente & ~esperado).any():
                    motivo += ' También hay ausencias con una matriculación; la causa no está documentada.'
                    decision = 'Conservar la matrícula y el hueco; estas variables de composición no se usan para calcular clientes.'
            elif nombre == 'stock_mensual' and c == 'wholesale':
                esperado = d.t == 97; motivo = 'Facturación disponible desde A009 febrero.'
            elif nombre == 'codigos' and c in ['primer_mes', 'ultimo_mes']:
                esperado = (d.unidades == 0) & (d.en_tarifa_A013 == 1)
                motivo = 'Código nuevo de A013 que todavía no tiene ventas.'
            elif nombre == 'segmentos' and c == 'vm_real_mas_180_eur':
                motivo = 'CSUV solo tiene cinco meses de historia; VM de más de 180 días no observado.'
                decision = 'Necesitará una hipótesis y sensibilidad en la etapa de costes, nunca cero por defecto.'
            filas.append({'archivo': nombre, 'columna': c, 'ausentes': int(ausente.sum()),
                          'explicados_por_regla': int((ausente & esperado).sum()),
                          'pendientes': int((ausente & ~esperado).sum()),
                          'interpretacion': motivo, 'decision': decision})
    return pd.DataFrame(filas)

def balance_stock(ventas, stock):
    """Salidas físicas observadas: matriculaciones, incluidas las automatriculaciones."""
    retail = ventas.groupby(['t', 'segmento'], as_index=False).unidades.sum().rename(columns={'unidades': 'retail'})
    d = stock.sort_values(['segmento', 't']).merge(retail, on=['t', 'segmento'], validate='one_to_one')
    d['stock_anterior'] = d.groupby('segmento').stock_total.shift()
    d['stock_esperado'] = d.stock_anterior + d.wholesale - d.retail
    d['diferencia'] = d.stock_total - d.stock_esperado
    d['estado'] = np.select([d.stock_esperado.isna(), d.diferencia == 0, d.diferencia.abs() <= 2],
                            ['Sin mes anterior', 'Cierra', 'Diferencia de 1 o 2 coches'], default='Revisar diferencia mayor')
    return d
