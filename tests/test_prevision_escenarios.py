"""Reglas críticas: futuro inaccesible, incertidumbre conjunta y demanda separada de oferta."""
import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
from src.prevision import (cargar_historia, proyectar_macro, ajustar_mercado, estimar_cuota,
                           seleccionar, metricas, SEGMENTOS)
from src.escenarios import (cargar_configuracion, construir_escenarios, preparar_residuos,
                           generar_factores, simular_demanda, resumir_simulaciones,
                           entradas_coste_suministro)
from src.rutas import RESULTADOS, leer_csv


class PrevisionEscenariosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ventas, cls.mercado, cls.macro = cargar_historia()
        cls.config = cargar_configuracion()

    def test_proyeccion_no_lee_economia_futura(self):
        m = self.macro
        esperado = proyectar_macro(m, 132, m[m.t.between(133, 144)])
        alterado = m.copy()
        for c in ['paro', 'viajeros_var', 'confianza_consumidor', 'caida_tras_adelanto']:
            alterado.loc[alterado.t > 132, c] = 99999
        pd.testing.assert_frame_equal(esperado, proyectar_macro(alterado, 132, alterado[alterado.t > 132]))

    def test_rechaza_calendario_desalineado(self):
        cal = self.macro[self.macro.t > 132].copy()
        cal.loc[cal.index[0], 't'] = 145
        with self.assertRaises(ValueError): proyectar_macro(self.macro, 132, cal)

    def test_pipeline_y_ajuste_solo_ven_pasado(self):
        original = ajustar_mercado(self.mercado, self.macro, 'MSUV', 'PARTICULARES', 132, 10)
        mercado = self.mercado.copy(); mercado.loc[mercado.t > 132, 'unidades'] = 999999
        macro = self.macro.copy(); macro.loc[macro.t > 132, 'paro'] = 999
        nuevo = ajustar_mercado(mercado, macro, 'MSUV', 'PARTICULARES', 132, 10)
        fut = proyectar_macro(self.macro, 132, self.macro[self.macro.t > 132])
        np.testing.assert_allclose(original.predecir(fut), nuevo.predecir(fut))
        scaler = original.pipeline.named_steps['preparar'].named_transformers_['numericas']
        self.assertEqual(int(scaler.n_samples_seen_), 132)
        self.assertAlmostEqual(scaler.mean_[0], self.macro[self.macro.t <= 132].paro.mean())
        betas = original.coeficientes()
        self.assertLessEqual(betas['paro'], 1e-9)
        self.assertGreaterEqual(betas['confianza_consumidor'], -1e-9)

    def test_cuota_no_usa_lanzamiento_futuro_ni_adelanto(self):
        q, n, _ = estimar_cuota(self.ventas, self.mercado, self.macro, 'CSUV', 'PARTICULARES', 132, 12)
        self.assertEqual((q, n), (0, 0))
        q, n, _ = estimar_cuota(self.ventas, self.mercado, self.macro, 'CSUV', 'PARTICULARES', 144, 12)
        self.assertEqual(n, 3)
        alterada = self.ventas.copy()
        alterada.loc[alterada.t > 142, 'clientes_observados'] *= 10
        q2, _, _ = estimar_cuota(alterada, self.mercado, self.macro, 'CSUV', 'PARTICULARES', 144, 12)
        self.assertEqual(q, q2)

    def test_seleccion_no_puede_consultar_A012(self):
        base = pd.DataFrame([{'t': t, 'segmento': 'B', 'alpha': a, 'ventana': 12,
                              'real': 100, 'prevision': p, 'ingenua_estacional': 95, 'validable': True}
                             for t in [121, 133] for a, p in [(1., 90.), (10., 99.)]])
        a = seleccionar(base, hasta=132)
        base.loc[base.t == 133, ['real', 'prevision']] = 1e9
        pd.testing.assert_frame_equal(a, seleccionar(base, hasta=132))
        self.assertEqual(a.iloc[0].alpha, 10.)

    def test_metricas_no_incluyen_CSUV_en_total_comparable(self):
        d = pd.DataFrame([{'t': t, 'segmento': s, 'real': 10 + t,
                           'prevision': 11 + t, 'ingenua_estacional': 12 + t}
                          for t in range(1, 13) for s in SEGMENTOS])
        a = metricas(d)
        d.loc[d.segmento == 'CSUV', ['real', 'prevision']] *= 1000
        pd.testing.assert_frame_equal(a, metricas(d))

    def cargar_resultados(self):
        ruta = RESULTADOS / '03_prevision_validacion'
        return {n: pd.read_csv(ruta / f'{n}.csv') for n in
                ['prevision_base_A013', 'macro_base_A013', 'coeficientes_A013', 'errores_fuera_muestra']}

    def test_costes_y_fabrica_no_reducen_demanda(self):
        d = self.cargar_resultados(); fichas = leer_csv('segmentos.csv')
        a, _ = construir_escenarios(d['prevision_base_A013'], d['macro_base_A013'], d['coeficientes_A013'], fichas, self.config)
        config = deepcopy(self.config)
        config['escenarios']['Pesimista']['arancel_adicional_pct_pff'] = 0.10
        config['escenarios']['Pesimista']['segmento_parada'] = 'B'
        b, _ = construir_escenarios(d['prevision_base_A013'], d['macro_base_A013'], d['coeficientes_A013'], fichas, config)
        pd.testing.assert_frame_equal(a, b)
        central = a[a.escenario == 'Central'].sort_values(['t', 'segmento', 'canal'])
        base = d['prevision_base_A013'].sort_values(['t', 'segmento', 'canal'])
        np.testing.assert_allclose(central.demanda_media, base.prevision)
        self.assertTrue((a.demanda_media >= 0).all())
        self.assertTrue((a.demanda_media <= a.mercado_escenario + 1e-9).all())
        _, suministro = entradas_coste_suministro(fichas, config)
        parada = suministro[suministro.factor_produccion == 0].iloc[0]
        self.assertEqual(parada.t_llegada - parada.t_fabricacion, 3)

    def test_bootstrap_reproducible_y_total_calculado_conjunto(self):
        d = self.cargar_resultados()
        residuos = preparar_residuos(d['errores_fuera_muestra'])
        a = generar_factores(residuos, 1000, 42)
        b = generar_factores(residuos, 1000, 42)
        np.testing.assert_array_equal(a, b)
        # Índices comunes preservan exactamente los ratios de dos columnas idénticas.
        copia = residuos.copy(); copia['C'] = copia['B']
        iguales = generar_factores(copia, 1000, 42)
        np.testing.assert_array_equal(iguales[:, :, 0], iguales[:, :, 1])
        p, _ = construir_escenarios(d['prevision_base_A013'], d['macro_base_A013'], d['coeficientes_A013'],
                                    leer_csv('segmentos.csv'), self.config)
        sims = simular_demanda(p, a)
        self.assertEqual(sims.shape, (3, 1000, 12, 7))
        self.assertTrue(np.issubdtype(sims.dtype, np.integer))
        self.assertGreaterEqual(sims.min(), 0)
        self.assertTrue((sims[0] <= sims[1]).all())
        self.assertTrue((sims[1] <= sims[2]).all())
        anual, _ = resumir_simulaciones(sims, p)
        total = anual[(anual.escenario == 'Central') & (anual.segmento == 'TOTAL')].iloc[0]
        self.assertEqual(total.p05, np.quantile(sims[1].sum(axis=(1, 2)), .05))
        self.assertTrue((anual.p05 <= anual.p50).all() and (anual.p50 <= anual.p95).all())


if __name__ == '__main__': unittest.main()
