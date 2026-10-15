"""Pruebas físicas y de información: tres meses, FIFO, ventas censuradas y edad."""
import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
from src.prevision import SEGMENTOS
from src.escenarios import ORDEN
from src.produccion import cargar_politica, pedido_a_tres_meses, plan_central, seguridad
from src.simulacion import (consumir_fifo, exposiciones_edad, cohortes_iniciales,
                            simular_trayectorias, resultados_anuales)


class LogisticaTest(unittest.TestCase):
    def setUp(self):
        self.p = cargar_politica()
        self.p['fraccion_transfer_deficit'] = 0.0
        inicial = pd.DataFrame({'segmento': SEGMENTOS, 'stock_total': 50,
                                 's_0_90': 50, 's_91_180': 0, 's_mas_180': 0})
        media = np.full((15, 7), 10.)
        llegadas, _, _ = plan_central(media, np.full((2, 7), 10), np.full(7, 50), np.zeros(7))
        self.plan = {'media': media, 'llegadas': llegadas, 'ss': np.zeros(7, dtype=int),
                     'stock_inicial': inicial,
                     'capacidad': pd.DataFrame({'segmento': SEGMENTOS, 'capacidad_nominal_supuesta': 1000})}
        self.suministro = pd.DataFrame([{'escenario': e, 'segmento': s, 'mes_fabricacion': m,
            't_fabricacion': 144+m, 't_llegada': 147+m, 'factor_produccion': 1., 'factor_capacidad': 1.}
            for e in ORDEN for s in SEGMENTOS for m in range(1, 13)])
        self.trans = pd.DataFrame([{'escenario': e, 'segmento': s, 'transfer_permitido': True}
                                   for e in ORDEN for s in SEGMENTOS])
        self.d = np.full((3, 2, 12, 7), 10, dtype=np.int64)

    def sim(self, d=None, p=None, suministro=None):
        return simular_trayectorias(self.d if d is None else d, self.plan, self.p if p is None else p,
            self.suministro if suministro is None else suministro, self.trans)

    def test_ejemplo_tabla_3_del_caso(self):
        pedido, proyectado = pedido_a_tres_meses(np.array([536]), np.array([[700], [700]]),
                                                np.array([[550], [600]]), np.array([850]), np.array([222]))
        self.assertEqual(proyectado[0], 786)
        self.assertEqual(pedido[0], 286)

    def test_cero_demanda_y_cero_stock_no_fabrican(self):
        l, p, _ = plan_central(np.zeros((15, 7)), np.zeros((2, 7)), np.zeros(7), np.zeros(7))
        self.assertEqual(l.sum(), 0)
        self.assertEqual(p.stock_cierre_esperado.sum(), 0)

    def test_seguridad_recoge_error_persistente(self):
        p = dict(self.p)
        media = np.full((12, 7), 100.)
        desviaciones = np.linspace(-40, 40, 1000)
        replicas = np.broadcast_to(media, (1000, 12, 7)) + desviaciones[:, None, None]
        s = seguridad(media, replicas, p)
        self.assertTrue((s.cuantil_error_tres_meses > s.formula_independiente).all())
        self.assertTrue((s.stock_seguridad >= s.cuantil_error_tres_meses).all())

    def test_FIFO_vende_el_coche_viejo_antes_del_nuevo(self):
        cantidades = np.array([[[2, 3, 5]]])
        restante, vendido = consumir_fifo(cantidades, np.array([[4]]))
        np.testing.assert_array_equal(vendido, [[[2, 2, 0]]])
        np.testing.assert_array_equal(restante, [[[0, 1, 5]]])

    def test_dias_coche_y_vencimientos_sin_recargar_pasado(self):
        # Car 1: edad 75, vendido día 15. Car 2: edad 165, no vendido.
        # Car 3: edad 195, ya vencido en A012: cero nuevos días de crédito.
        v = np.array([[[1, 0, 0]]]); r = np.array([[[0, 1, 1]]])
        x = exposiciones_edad(v, r, np.array([75, 165, 195]))
        self.assertEqual(x['dias_coche_0_90'][0, 0], 15)
        self.assertEqual(x['dias_coche_91_180'][0, 0], 15)
        self.assertEqual(x['vencimientos_credito_nuevos'][0, 0], 1)
        self.assertEqual(x['ventas_0_90'][0, 0], 1)
        # Venta exactamente en día 180: aún no exige pago en día 181.
        y = exposiciones_edad(np.array([[[1]]]), np.array([[[0]]]), np.array([165]))
        self.assertEqual(y['vencimientos_credito_nuevos'][0, 0], 0)

    def test_cohortes_conservan_tramos_en_tres_hipotesis(self):
        stock = self.plan['stock_inicial'].copy()
        stock['s_0_90'] = 7; stock['s_91_180'] = 5; stock['s_mas_180'] = 3; stock['stock_total'] = 15
        for modo in ['centros', 'jovenes', 'mayores']:
            q, a = cohortes_iniciales(stock, modo)
            np.testing.assert_array_equal(q.sum(axis=1), np.full(7, 15))
            self.assertTrue((a[:4] > 90).all() and (a[4:] <= 90).all())

    def test_no_se_puede_ver_la_demanda_del_futuro(self):
        a = self.sim()
        d = self.d.copy(); d[:, :, 7:] = 1000
        b = self.sim(d=d)
        for c in a: np.testing.assert_array_equal(a[c][:, :, :7], b[c][:, :, :7], err_msg=c)

    def test_revision_solo_afecta_llegadas_tres_meses_despues(self):
        a = self.sim()
        d = self.d.copy(); d[:, :, 0] = 100
        b = self.sim(d=d)
        np.testing.assert_array_equal(a['llegadas'][:, :, :3], b['llegadas'][:, :, :3])
        self.assertTrue((a['llegadas'][:, :, 3] != b['llegadas'][:, :, 3]).any())
        np.testing.assert_array_equal(b['llegadas'][:, :, 3:], b['fabricado'][:, :, :9])

    def test_revision_con_ventas_no_observa_deficit_oculto(self):
        d = self.d.copy(); d[:, :, 0] = 100
        a = self.sim(d=d)
        d[:, :, 0] = 1000
        b = self.sim(d=d)
        np.testing.assert_array_equal(a['fabricar_solicitado'][:, :, 0], b['fabricar_solicitado'][:, :, 0])
        np.testing.assert_array_equal(a['observacion_revision'][:, :, 0], b['observacion_revision'][:, :, 0])
        self.assertTrue((b['ventas_perdidas'][:, :, 0] > a['ventas_perdidas'][:, :, 0]).all())

    def test_parada_de_junio_llega_en_septiembre(self):
        self.plan['stock_inicial'][['stock_total', 's_0_90']] = 0
        self.plan['llegadas'][:] = 10
        a = self.sim()
        suministro = self.suministro.copy()
        parada = (suministro.escenario == 'Pesimista') & (suministro.segmento == 'MSUV') & (suministro.mes_fabricacion == 6)
        suministro.loc[parada, 'factor_produccion'] = 0.
        b = self.sim(suministro=suministro)
        j = SEGMENTOS.index('MSUV')
        np.testing.assert_array_equal(a['llegadas'][0, :, :8], b['llegadas'][0, :, :8])
        self.assertTrue((b['llegadas'][0, :, 8, j] == 0).all())
        self.assertTrue((b['produccion_anulada'][0, :, 5, j] > 0).all())

    def test_transfer_no_sale_del_stock_y_total_no_suma_ratios(self):
        d = self.d.copy(); d[:, :, 0] = 100
        p = dict(self.p); p['fraccion_transfer_deficit'] = .5
        out = self.sim(d=d, p=p)
        self.assertTrue((out['transfers'][:, :, 0] > 0).all())
        np.testing.assert_array_equal(out['stock_inicio']+out['llegadas']-out['ventas_locales'], out['stock_final'])
        a = resultados_anuales(out)
        total = a.query('segmento == "TOTAL"')
        self.assertTrue((total.meses_con_ventas_perdidas <= 12).all())
        np.testing.assert_allclose(total.servicio_unidades_pct, 100*total.ventas/total.demanda)

    def test_stock_medio_respeta_el_dia_de_venta(self):
        p = deepcopy(self.p); p['dia_venta'] = 10
        out = self.sim(p=p)
        # Enero: 60 coches antes de vender; 50 quedan desde día 10 hasta día 30.
        self.assertAlmostEqual(out['stock_medio'][0, 0, 0, 0], (60*10+50*20)/30)

    def test_A014_cambia_pedidos_finales_sin_alterar_ventas_A013(self):
        a = self.sim()
        self.plan['media'][12:] = 30
        b = self.sim()
        for c in ['stock_final', 'ventas', 'llegadas', 'ventas_perdidas']:
            np.testing.assert_array_equal(a[c], b[c])
        self.assertTrue((a['fabricar_solicitado'][:, :, 9:] != b['fabricar_solicitado'][:, :, 9:]).any())


if __name__ == '__main__': unittest.main()
