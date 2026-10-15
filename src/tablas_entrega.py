"""Exporta las tablas numéricas del resumen sin generar documentos o paquetes."""
import pandas as pd
from .rutas import RESULTADOS
from .prevision import SEGMENTOS


def exportar_tablas():
    plan = pd.read_csv(RESULTADOS / '05_produccion_stock/plan_central.csv')
    produccion = plan.pivot(index='mes', columns='segmento', values='fabricar_plan_inicial')
    produccion = produccion.reindex(columns=SEGMENTOS).astype(int)
    produccion['TOTAL'] = produccion.sum(axis=1)
    completa = pd.read_csv(RESULTADOS / '07_costes_margen/tabla_5.csv')
    columnas = ['escenario', 'segmento', 'ventas', 'fabricado', 'stock_medio',
                'stock_medio_meses_demanda', 'stock_final', 'stock_mas_90',
                'coste_stock_marca_eur', 'ventas_perdidas', 'oportunidad_ventas_perdidas_eur',
                'transfers', 'coste_transfers_eur', 'vm_eur', 'cvp_despues_vm_eur',
                'contribucion_marca_eur', 'servicio_unidades_pct',
                'intereses_concesionarios_eur', 'principal_vencido_nuevo_eur']
    tabla = completa[columnas].copy()
    assert len(produccion) == 12 and len(tabla) == 24
    salida = RESULTADOS / '09_entrega_defensa'
    salida.mkdir(parents=True, exist_ok=True)
    produccion.to_csv(salida / 'produccion_mensual_plan_inicial.csv')
    tabla.to_csv(salida / 'tabla_5_total_y_segmentos.csv', index=False)
    return produccion, tabla
