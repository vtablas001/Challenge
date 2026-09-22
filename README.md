# Marketing Mix Model Challenge

Este repositorio contiene un análisis reproducible de volumen semanal y una secuencia de modelos para separar la venta base del aporte asociado a medios. El notebook conserva la exploración y el razonamiento; `main.py` ejecuta la versión modular de punta a punta.

## Diseño temporal

La variable dependiente es `log1p(volume_hl)` y las predicciones se retransforman a hectolitros mediante smearing de Duan.

- Calibración: 2023-01-07 a 2024-11-23.
- Validación: 2024-11-30 a 2025-05-17.
- Prueba final: 2025-05-24 a 2025-12-27.

La calibración conjunta usa 2,000 ensayos Optuna TPE y tres ventanas expansivas de ocho semanas dentro de validación. Ninguna decisión usa prueba. Los 14 canales entran simultáneamente y sus coeficientes se restringen a ser no negativos; el signo no es un criterio de selección.

## Resultados finales

| Paso | Modelo | R2 prueba | RMSE hL | MAE hL | WAPE | MASE agregado | RMSE backtest |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Estacional ingenuo | 0.7841 | 1,005.11 | 299.69 | 0.3984 | 0.7403 | 988.36 |
| 1 | Venta base | 0.6287 | 1,318.07 | 498.57 | 0.6628 | 1.2316 | 1,297.43 |
| 2 | Venta base + lag52 | 0.8620 | 803.62 | 308.70 | 0.4104 | 0.7626 | 790.78 |
| 3 | MMM estándar | 0.6473 | 1,284.57 | 482.33 | 0.6412 | 1.1915 | 1,258.93 |
| 4 | MMM calibrado conjunto | 0.6393 | 1,299.07 | 489.25 | 0.6504 | 1.2086 | 1,276.41 |
| 5 | MMM calibrado + lag52 | 0.8574 | 816.95 | 311.56 | 0.4142 | 0.7697 | 804.89 |

El MMM estándar pierde 2.91% de RMSE de validación frente al calibrado, dentro de la tolerancia predefinida de 5%; por parsimonia se prefiere el Paso 3 como especificación principal de medios. El Paso 4 se mantiene para la atribución calibrada pedida y el Paso 5 para pronóstico. A4, con efectos fijos por serie, se reporta como robustez y no altera una decisión fijada antes de observar prueba.

Marca C y Subcanal B son las categorías de referencia de los efectos fijos. Los coeficientes de las demás marcas y subcanales se interpretan como diferencias frente a esas categorías. Los medios son predictores continuos y conservan un coeficiente propio; TV local se usa como comparador de negocio en los índices de eficiencia y ROI, donde toma el valor 1.00.

Las contribuciones y cocientes de retorno son atribuciones condicionadas al modelo y al gasto histórico. No representan efectos causales ni el retorno marginal de aumentar presupuesto. Una decisión de inversión requiere ubicar el adstock reciente respecto de `gamma`, evaluar la pendiente local de Hill, incorporar costos e incertidumbre de los parámetros y considerar variables omitidas o asignación endógena del gasto. Por ello, los canales con asociaciones favorables se tratan como candidatos para pruebas incrementales con grupos de control.

El rezago anual tiene cinco observaciones faltantes en 2023, concentradas en una serie (Marca B, Región G, Subcanal A). Los modelos con `lag52` excluyen esas filas sin imputarlas.

## Ejecución

```powershell
uv sync --dev
uv run pytest
uv run python main.py --n-trials 2000
```

Optuna conserva el estudio local en `outputs/optuna_mmm.db`, archivo excluido de Git. Una ejecución interrumpida reanuda hasta completar el presupuesto solicitado.

## Estructura

- `mmm/data.py`: carga, integración, fechas y rezago anual.
- `mmm/transformations.py`: adstock geométrico, Hill y grillas.
- `mmm/models.py`: diseño del panel, restricciones y smearing.
- `mmm/calibration.py`: búsqueda conjunta y validación temporal.
- `mmm/evaluation.py`: métricas, MASE, backtesting y errores por segmento.
- `mmm/pipeline.py`: secuencia, atribución, sensibilidad y salidas.
- `tests/`: controles automáticos de fechas, fuga, gamma y lag52.

## Salidas

- `outputs/tabla_modelos.csv`: prueba y backtesting de modelos principales y apéndices.
- `outputs/contribuciones_roi.csv`: atribución modelada y cocientes históricos por canal y año; no equivalen a ROI causal o marginal.
- `outputs/sensibilidad_roi.csv`: variación de la atribución modelada entre las 18 combinaciones por canal.
- `outputs/hiperparametros_paso4.json`: parámetros, gammas resueltos, semilla y ventanas.
- `outputs/errores_por_segmento.csv`: error por marca y subcanal.
- `outputs/figures/`: RMSE, contribuciones y curvas de respuesta.
