"""Pruebas de las reglas que pueden cambiar las conclusiones de las etapas 01 y 02."""
import unittest
import numpy as np
import pandas as pd
from src.historia import preparar_demanda, agregar_mensual, comparar_anios, marcar_stock, referencias_estacionales, EVENTOS
from src.calidad import auditar, interpretar_ausencias
from src.rutas import leer_csv

class ReglasHistoricas(unittest.TestCase):
    def ventas(self):
        return pd.DataFrame({'t': [133, 133, 133], 'anio': ['A012']*3, 'mes': [1]*3,
            'segmento': ['B']*3, 'canal': ['PARTICULARES', 'EMPRESAS_RENTING', 'ALQUILADORES'],
            'unidades': [10, 5, 3], 'u_automatriculacion': [2, 1, 0],
            'u_venta_especial': [4, 2, 0], 'renovacion_gama': [0]*3})

    def macro(self):
        return pd.DataFrame({'t': [133], **{c: [0] for c in EVENTOS}})

    def test_no_se_restan_ventas_especiales_y_se_conservan_canales(self):
        d = preparar_demanda(self.ventas(), self.macro())
        self.assertEqual(len(d), 3)
        self.assertEqual(int(d.clientes_observados.sum()), 15)
        self.assertEqual(int(d.unidades.sum()), 18)

    def test_union_rechaza_sucesos_duplicados_y_meses_ausentes(self):
        with self.assertRaises(ValueError):
            preparar_demanda(self.ventas(), pd.concat([self.macro(), self.macro()]))
        with self.assertRaises(ValueError):
            preparar_demanda(self.ventas(), self.macro().assign(t=132))

    def test_automatriculaciones_no_pueden_superar_unidades(self):
        v = self.ventas(); v.loc[0, 'u_automatriculacion'] = 11
        with self.assertRaises(ValueError): preparar_demanda(v, self.macro())

    def test_stock_no_se_triplica_por_los_canales(self):
        d = preparar_demanda(self.ventas(), self.macro())
        mensual = agregar_mensual(d)
        st = pd.DataFrame({'t': [133], 'segmento': ['B'], 'stock_total': [2], 'wholesale': [4]})
        r = marcar_stock(mensual, st)
        self.assertEqual(len(r), 1)
        self.assertEqual(r.stock_total.iloc[0], 2)
        self.assertAlmostEqual(r.cobertura_cierre_clientes_mes.iloc[0], 2/15)
        with self.assertRaises(ValueError): marcar_stock(pd.concat([mensual, mensual]), st)

    def test_stock_ausente_no_es_cero_y_cero_sin_clientes_no_es_ruptura(self):
        m = pd.DataFrame({'t': [96, 97], 'segmento': ['B', 'B'], 'clientes_observados': [10, 0]})
        st = pd.DataFrame({'t': [97], 'segmento': ['B'], 'stock_total': [0], 'wholesale': [0]})
        d = marcar_stock(m, st)
        self.assertTrue(pd.isna(d.stock_total.iloc[0]))
        self.assertFalse(d.senal_stock_bajo.any())
        self.assertFalse(d.stock_cero_con_ventas.any())

    def test_lanzamiento_sin_base_no_tiene_crecimiento_porcentual(self):
        d = pd.DataFrame({'anio': ['A011', 'A012'], 'segmento': ['CSUV', 'CSUV'],
                          'unidades': [0, 20], 'u_automatriculacion': [0, 5], 'clientes_observados': [0, 15]})
        r = comparar_anios(d, 'segmento')
        self.assertTrue(pd.isna(r.crecimiento_clientes_pct.iloc[0]))
        self.assertEqual(r.variacion_clientes.iloc[0], 15)

    def test_evento_no_calibra_su_propia_referencia(self):
        filas = []
        for anio, factor in [('A009', 1), ('A010', 1), ('A012', 2)]:
            for mes in range(1, 13):
                clientes = factor*(100+mes) + (50 if anio == 'A012' and mes >= 11 else 0)
                filas.append({'anio': anio, 'mes': mes, 'segmento': 'B', 'unidades': clientes,
                              'clientes_observados': clientes})
        d = pd.DataFrame(filas)
        referencias, _ = referencias_estacionales(d)
        np.testing.assert_allclose(referencias.factor_nivel, 2)
        np.testing.assert_allclose(referencias.diferencia_coches, 100)
        cambio = d.copy()
        cambio.loc[(cambio.anio == 'A012') & (cambio.mes >= 11), 'clientes_observados'] += 500
        otras, _ = referencias_estacionales(cambio)
        np.testing.assert_allclose(otras.factor_nivel, referencias.factor_nivel)
        with self.assertRaises(ValueError): referencias_estacionales(d, meses_nivel=(11, 12))

class RevisionDeFuentes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tablas = {n: leer_csv(n + '.csv') for n in ['ventas_mensual', 'mercado_mensual',
            'stock_mensual', 'macro_mensual', 'segmentos', 'codigos', 'calendario_A013']}

    def test_generaciones_reutilizadas_y_cvp_negativo_son_validos(self):
        c = self.tablas['codigos']; s = self.tablas['segmentos']
        self.assertTrue(c.duplicated(['segmento', 'codigo_modelo']).any())
        self.assertTrue((s.cvp_tarifa_eur < 0).any())
        self.assertTrue((auditar(self.tablas).estado == 'OK').all())

    def test_una_fraccion_fuera_de_rango_es_detectada(self):
        tablas = {n: d.copy() for n, d in self.tablas.items()}
        tablas['mercado_mensual'].loc[0, 'pct_diesel'] = 9
        r = auditar(tablas)
        self.assertEqual(r.loc[r.control == 'Fracción mercado_mensual.pct_diesel', 'incidencias'].iloc[0], 1)

    def test_las_ausencias_economicas_no_se_rellenan(self):
        originales = self.tablas['segmentos'].copy(deep=True)
        r = interpretar_ausencias(self.tablas)
        self.assertEqual(r.loc[r.columna == 'vm_real_mas_180_eur', 'pendientes'].iloc[0], 1)
        pd.testing.assert_frame_equal(originales, self.tablas['segmentos'])

if __name__ == '__main__': unittest.main()
