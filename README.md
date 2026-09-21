# Marketing Mix Model Challenge

Este repositorio contiene un analisis reproducible de volumen semanal y una secuencia de modelos para separar la venta base del aporte asociado a medios. El notebook conserva la exploracion y el razonamiento; `main.py` ejecuta la version modular de punta a punta.

## Diseno temporal

La variable dependiente es `log1p(volume_hl)` y las predicciones se retransfoman a hectolitros mediante smearing de Duan.

- Calibracion: 2023-01-07 a 2024-11-23.
- Validacion: 2024-11-30 a 2025-05-17.
- Prueba final: 2025-05-24 a 2025-12-27.

La calibracion conjunta usa 2,000 ensayos Optuna TPE y tres ventanas expansivas de ocho semanas dentro de validacion. Ninguna decision usa prueba. Los 14 canales entran simultaneamente y sus coeficientes se restringen a ser no negativos; el signo no es un criterio de seleccion.

## Resultados finales

| Paso | Modelo | R2 prueba | RMSE hL | MAE hL | WAPE | MASE agregado | RMSE backtest |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Estacional ingenuo | 0.7841 | 1,005.11 | 299.69 | 0.3984 | 0.7403 | 988.36 |
| 1 | Venta base | 0.5764 | 1,407.95 | 465.35 | 0.6186 | 1.1496 | 1,376.49 |
| 2 | Venta base + lag52 | 0.7748 | 1,026.45 | 309.96 | 0.4120 | 0.7657 | 947.11 |
| 3 | MMM estandar | 0.5926 | 1,380.66 | 462.12 | 0.6143 | 1.1416 | 1,353.36 |
| 4 | MMM calibrado conjunto | 0.6011 | 1,366.21 | 466.54 | 0.6202 | 1.1525 | 1,343.97 |
| 5 | MMM calibrado + lag52 | 0.8447 | 852.39 | 288.49 | 0.3835 | 0.7127 | 823.12 |

El MMM estandar pierde 1.39% de RMSE de validacion frente al calibrado, dentro de la tolerancia predefinida de 5%; por parsimonia se prefiere el Paso 3 como especificacion principal de medios. El Paso 4 se mantiene para la atribucion calibrada pedida y el Paso 5 para pronostico. A4, con efectos fijos por serie, se reporta como robustez y no altera una decision fijada antes de observar prueba.

El rezago anual tiene cinco observaciones faltantes en 2023, concentradas en una serie (Marca B, Region G, Subcanal A). Los modelos con `lag52` excluyen esas filas sin imputarlas.

## Ejecucion

```powershell
uv sync --dev
uv run pytest
uv run python main.py --n-trials 2000
```

Optuna conserva el estudio local en `outputs/optuna_mmm.db`, archivo excluido de Git. Una ejecucion interrumpida reanuda hasta completar el presupuesto solicitado.

## Estructura

- `mmm/data.py`: carga, integracion, fechas y rezago anual.
- `mmm/transformations.py`: adstock geometrico, Hill y grillas.
- `mmm/models.py`: diseno del panel, restricciones y smearing.
- `mmm/calibration.py`: busqueda conjunta y validacion temporal.
- `mmm/evaluation.py`: metricas, MASE, backtesting y errores por segmento.
- `mmm/pipeline.py`: secuencia, atribucion, sensibilidad y salidas.
- `tests/`: controles automaticos de fechas, fuga, gamma y lag52.

## Salidas

- `outputs/tabla_modelos.csv`: prueba y backtesting de modelos principales y anexos.
- `outputs/contribuciones_roi.csv`: contribucion y retorno por canal y ano.
- `outputs/sensibilidad_roi.csv`: rango de retorno en las 18 combinaciones por canal.
- `outputs/hiperparametros_paso4.json`: parametros, gammas resueltos, semilla y ventanas.
- `outputs/errores_por_segmento.csv`: error por marca y subcanal.
- `outputs/figures/`: RMSE, contribuciones y curvas de respuesta.

A5 reproduce el procedimiento individual anterior unicamente como comparacion historica; no participa en la seleccion del modelo final.
