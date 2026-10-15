# Kaido Motors España

Trabajo de la asignatura **Simulación de Sistemas Logísticos**, basado en el caso Kaido Motors España.

El objetivo es proponer un plan de fabricación y stock para A013, teniendo en cuenta que la demanda es incierta y que los coches fabricados tardan tres meses en llegar. Las decisiones deben equilibrar las ventas que se pierden por falta de disponibilidad y los costes de acumular coches que envejecen en la red.

El caso se sitúa al finalizar A012. Los años anteriores permiten analizar el negocio y comprobar el modelo; A013 es el año futuro que debemos planificar.

## Trabajo realizado

La solución aborda las cinco preguntas del caso:

1. Analizar las compras observadas de clientes, separando automatriculaciones y estudiando lanzamientos, adelantos de compras y posibles restricciones de stock.
2. Construir una previsión mensual de demanda por segmento, considerando economía, calendario y producto, y comprobarla con A012.
3. Simular escenarios pesimista, central y optimista con incertidumbre.
4. Convertir la demanda en un plan de producción y simular mensualmente llegadas, ventas y stock.
5. Calcular costes y margen, comparar políticas y formular recomendaciones.

El modelo combina una previsión del mercado mediante regresiones Ridge con una estimación de la cuota de Kaido. Sus parámetros se seleccionan con información anterior a A012. Después de comprobar el resultado en ese año, se reajusta con el histórico disponible para preparar A013.

La simulación utiliza **1.000 réplicas por escenario**. Los tres escenarios parten del mismo plan inicial central y permiten revisar las decisiones respetando el plazo de tres meses. Se sigue la antigüedad del inventario y se venden primero los coches más antiguos, mediante FIFO.

## Organización del repositorio

```text
.
├── notebooks/        Análisis explicado y ejecutable
├── src/              Funciones del modelo y de la simulación
├── data/             Siete CSV originales del caso
├── configuracion/    Hipótesis de escenarios, producción y costes
├── resultados/       Tablas, gráficos y trayectorias generadas
├── documentos/       Resumen ejecutivo y respuestas desarrolladas
├── tests/            Pruebas de las reglas del modelo
└── requirements.txt  Dependencias de Python
```

### Notebooks

Se ejecutan en el siguiente orden:

| Notebook | Contenido |
|---|---|
| [01 · Datos y demanda histórica](notebooks/01_datos_y_demanda_historica.ipynb) | Comprobación de las fuentes, compras de clientes y análisis del crecimiento de A012. |
| [02 · Modelo y validación](notebooks/02_modelo_y_validacion.ipynb) | Construcción del modelo, validación temporal, comparación con una referencia estacional y previsión de A013. |
| [03 · Escenarios, producción y simulación](notebooks/03_escenarios_produccion_y_simulacion.ipynb) | Escenarios, incertidumbre, stock inicial, llegadas comprometidas, fabricación y movimiento mensual de coches. |
| [04 · Costes y políticas](notebooks/04_costes_y_politicas.ipynb) | Valoración económica, comparación de políticas, recomendaciones y exportación de las tablas del resumen. |

Los notebooks incluyen explicaciones, código y resultados ejecutados. Las funciones compartidas están en `src/`.

Las subcarpetas de `resultados/` conservan la numeración de los análisis originales, aunque estos se hayan agrupado en cuatro notebooks.

### Documentos

En `documentos/` se incluyen:

- **Resumen ejecutivo:** propuesta de fabricación mensual y resultados de ventas, stock, costes y margen en los tres escenarios.
- **Respuestas a las cinco preguntas:** explicación ampliada de los resultados, el método y sus limitaciones.

Ambos documentos están disponibles en PDF y Word.

### Datos e hipótesis

`data/` contiene los siete CSV originales: ventas, mercado, stock, variables económicas, segmentos, códigos de producto y calendario de A013.

Las hipótesis editables están en:

| Archivo | Contenido |
|---|---|
| `configuracion/escenarios_A013.json` | Escenarios e incertidumbre de la demanda. |
| `configuracion/politica_stock.json` | Política logística, stock de seguridad y revisión de pedidos. |
| `configuracion/costes.json` | Parámetros de valoración económica. |

Los datos originales se conservan sin modificar. La lectura distingue las celdas vacías del código de producto `NA`, que es un valor válido.

## Cómo consultar el trabajo

Para conocer la propuesta, comienza por el resumen ejecutivo de `documentos/`.

Para entender cómo se obtiene, recorre los cuatro notebooks en orden. Sus resultados ya están guardados, por lo que pueden consultarse sin recalcular el modelo.

## Cómo ejecutar el proyecto

La ejecución se ha comprobado con **Python 3.12**. Abre una terminal en la raíz del repositorio.

### 1. Preparar el entorno

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, la activación del entorno se realiza con:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Ejecutar los notebooks

Para abrirlos y ejecutarlos manualmente:

```bash
python -m jupyter lab
```

Selecciona el entorno de Python donde has instalado las dependencias y ejecuta todas las celdas, siguiendo el orden 01–04.

Para ejecutar automáticamente el recorrido completo:

```bash
python -m src.ejecutar_notebooks
```

El ejecutor utiliza un kernel independiente para cada notebook, guarda sus salidas y registra la ejecución en `resultados/verificacion_notebooks.json`.

La ejecución actualiza los resultados y los notebooks. **Los documentos PDF y Word no se regeneran automáticamente**: deben revisarse si cambian las cifras o las hipótesis.

### 3. Ejecutar las pruebas

```bash
python -m unittest discover -s tests -v
```

Se incluyen 40 pruebas sobre reglas relevantes: separación de automatriculaciones, uso de información disponible en cada fecha, balances de stock, plazo de llegada, FIFO, transfers, costes y agregación de resultados.

## Interpretación y limitaciones

- La demanda prevista representa compras potenciales; las ventas simuladas dependen de la disponibilidad. Un exceso de stock en un segmento no cubre automáticamente la demanda de otro.
- **CVP** es el beneficio bruto menos el gasto comercial, VM. La **contribución** incorpora además los costes de permanencia, transfers y ajustes económicos del escenario. No equivale a beneficio neto después de costes fijos e impuestos.
- Los rangos de simulación dependen del modelo y de las hipótesis. No se asignan probabilidades a los tres escenarios.
- El modelo conserva una limitación explícita: en A012 obtiene un WAPE comparable del **14,96 %**, frente al **12,32 %** de la referencia estacional. Por tanto, la previsión sigue siendo provisional.
- Las llegadas iniciales, la capacidad y algunos costes requieren hipótesis porque no se dispone de toda la información operativa.

La solución permite reproducir y estudiar las consecuencias del plan propuesto. Las recomendaciones deben interpretarse junto con estas limitaciones.

## Material del caso

El enunciado y los datos son material docente del caso Kaido Motors, de Jacinto Velasco y Javier Goikoetxea. Su uso y distribución están sujetos a las condiciones indicadas por los autores.
