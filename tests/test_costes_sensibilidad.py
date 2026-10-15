import unittest
import numpy as np
import pandas as pd
from src.costes import cargar_costes, valorar, anuales_economicos
from src.sensibilidad import comparacion_pareada, ranking_robusto
from src.rutas import leer_csv
from src.prevision import SEGMENTOS
from src.escenarios import ORDEN


class CostesTests(unittest.TestCase):
    def setUp(self):
        self.c = cargar_costes(); self.f = leer_csv('segmentos.csv')
        self.out = {k: np.zeros((3, 2, 12, 7), dtype=int) for k in [
            'ventas', 'ventas_locales', 'transfers', 'ventas_perdidas', 'stock_inicio',
            'ventas_0_90', 'ventas_91_180', 'ventas_mas_180', 'dias_coche_0_90',
            'dias_coche_91_180', 'vencimientos_credito_nuevos']}
        self.entradas = pd.DataFrame([{'escenario': e, 'segmento': s,
            'arancel_adicional_eur_coche_proxy': 0., 'prima_co2_eur_coche': 0.,
            'ahorro_cambio_eur_coche_proxy': 0., 'coste_neto_adicional_eur_coche_proxy': 0.}
            for e in ORDEN for s in SEGMENTOS])
    def calcular(self): return valorar(self.out, self.f, self.entradas, self.c)
    def test_coche_fresco_msuv_a_60_dias(self):
        j = SEGMENTOS.index('MSUV')
        for k in ['ventas', 'ventas_locales', 'ventas_0_90']: self.out[k][0,0,0,j] = 1
        self.out['dias_coche_0_90'][0,0,0,j] = 60
        e = self.calcular()
        self.assertAlmostEqual(e['contribucion_marca_eur'][0,0,0,j], 7782-1895-120)
        self.assertAlmostEqual(e['intereses_concesionarios_eur'][0,0,0,j],25738*.01*60/365)
    def test_transfer_sin_stock_ni_segundo_vm(self):
        j = SEGMENTOS.index('MSUV')
        for k in ['ventas','transfers','ventas_0_90']: self.out[k][0,0,0,j] = 1
        e = self.calcular()
        self.assertAlmostEqual(e['contribucion_marca_eur'][0,0,0,j],7782*.8-1895)
        self.assertEqual(e['coste_stock_marca_eur'].sum(),0)
    def test_venta_perdida_no_resta_margen_realizado(self):
        self.out['ventas_perdidas'][:] = 10
        e = self.calcular()
        self.assertEqual(e['contribucion_marca_eur'].sum(),0)
        self.assertGreater(e['oportunidad_ventas_perdidas_eur'].sum(),0)
    def test_principal_es_caja_separada(self):
        self.out['vencimientos_credito_nuevos'][:] = 1
        e = self.calcular()
        self.assertEqual(e['contribucion_marca_eur'].sum(),0)
        self.assertGreater(e['principal_vencido_nuevo_eur'].sum(),0)
    def test_csuv_viejo_solo_afecta_ventas_viejas(self):
        j=SEGMENTOS.index('CSUV')
        self.out['ventas'][0,0,0,j]=self.out['ventas_locales'][0,0,0,j]=self.out['ventas_mas_180'][0,0,0,j]=2
        e=self.calcular()
        self.assertAlmostEqual(e['vm_eur'][0,0,0,j],2*2470*1.2)
        self.assertEqual(e['ventas_csuv_vm_imputado'].sum(),2)
    def test_stock_inicial_exento_arancel_pero_no_co2(self):
        j=SEGMENTOS.index('C')
        self.out['stock_inicio'][0,0,0,j]=3
        for k in ['ventas','ventas_locales','ventas_0_90']: self.out[k][0,0,:2,j]=2
        self.entradas['arancel_adicional_eur_coche_proxy']=100
        self.entradas['prima_co2_eur_coche']=200
        e=self.calcular()
        self.assertEqual(e['ventas_stock_A012'][0,0,:,j].sum(),3)
        self.assertEqual(e['arancel_adicional_eur'][0,0,:,j].sum(),100)
        self.assertEqual(e['prima_co2_eur'][0,0,:,j].sum(),800)
    def test_total_cuantiles_se_calcula_tras_sumar(self):
        e=self.calcular()
        e['contribucion_marca_eur'][0,0,0,0]=100
        e['contribucion_marca_eur'][0,1,0,1]=100
        a=anuales_economicos(e)
        self.assertTrue((a.query('escenario == "Pesimista" and segmento == "TOTAL"').contribucion_marca_eur==100).all())
    def test_delta_pareado_respeta_replica(self):
        base=pd.DataFrame({'escenario':['Central']*2,'segmento':['TOTAL']*2,'replica':[0,1],
            **{k:[10,20] for k in ['contribucion_marca_eur','ventas_perdidas','stock_medio','servicio_unidades_pct']}})
        cand=base.iloc[::-1].copy(); cand.contribucion_marca_eur+=5
        z=comparacion_pareada(cand,base)
        self.assertEqual(z.query('metrica == "contribucion_marca_eur"').delta_p05.iloc[0],5)
        with self.assertRaises(ValueError): comparacion_pareada(cand.iloc[:1],base)
    def test_ranking_no_promedia_escenarios(self):
        z=pd.DataFrame({'variante':['A']*3+['B']*3,'segmento':['TOTAL']*6,
            'contribucion_marca_eur':[1,100,100,2,3,4], 'contribucion_p05_eur':[0]*6,
            'servicio_unidades_pct':[96]*6,'stock_medio':[10]*6})
        self.assertEqual(ranking_robusto(z).iloc[0].variante,'B')

if __name__=='__main__': unittest.main()
