# Marketing Mix Model Challenge

Este repositorio contiene un análisis reproducible de volumen semanal y una secuencia de modelos para separar la venta base del aporte asociado a medios. El notebook conserva la exploración y el razonamiento; `main.py` ejecuta la versión modular de punta a punta.

## Diseño temporal

La variable dependiente es `log1p(volume_hl)` y las predicciones se retransforman a hectolitros mediante smearing de Duan.

- Calibración: 2023-01-07 a 2024-11-23.
- Validación: 2024-11-30 a 2025-05-17.
- Prueba final: 2025-05-24 a 2025-12-27.

La calibración conjunta usa 2,000 ensayos Optuna TPE y tres ventanas expansivas de ocho semanas dentro de validación. Ninguna decisión usa prueba. Los 14 canales entran simultáneamente y sus coeficientes se restringen a ser no negativos; el signo no es un criterio de selección.

### Estacionalidad anual

La especificación principal excluye seno y coseno anual. La decisión se tomó antes de consultar la prueba final, comparando tres modelos en las mismas ventanas de validación. Al retirarlos, el RMSE disminuyó 4.03% en venta base, 3.02% en el MMM estándar y 2.46% en el MMM calibrado con parámetros fijos. Aunque MAE y WAPE aumentaron cerca de 6%, se mantuvo la especificación más simple porque el criterio principal predefinido era RMSE y la dirección fue consistente en los tres modelos. Los indicadores de semana del año permanecen únicamente como robustez en A2 y A3.

## Resultados finales

| Paso | Modelo | R2 prueba | RMSE hL | MAE hL | WAPE | MASE agregado | RMSE backtest |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Estacional ingenuo | 0.7841 | 1,005.11 | 299.69 | 0.3984 | 0.7403 | 988.36 |
| 1 | Venta base | 0.5665 | 1,424.24 | 459.30 | 0.6106 | 1.1346 | 1,391.87 |
| 2 | Venta base + lag52 | 0.7699 | 1,037.73 | 310.65 | 0.4129 | 0.7674 | 952.08 |
| 3 | MMM estándar | 0.5775 | 1,406.06 | 449.42 | 0.5974 | 1.1102 | 1,359.51 |
| 4 | MMM calibrado conjunto | 0.5816 | 1,399.28 | 452.76 | 0.6019 | 1.1185 | 1,356.93 |
| 5 | MMM calibrado + lag52 | 0.8157 | 928.60 | 294.95 | 0.3921 | 0.7286 | 860.79 |

El MMM estándar pierde 2.12% de RMSE de validación frente al calibrado, dentro de la tolerancia predefinida de 5%; por parsimonia se prefiere el Paso 3 como especificación principal de medios. El Paso 4 se mantiene para la atribución calibrada pedida y el Paso 5 para pronóstico. A4, con efectos fijos por serie, se reporta como robustez y no altera una decisión fijada antes de observar prueba.

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
- `outputs/contribuciones_roi.csv`: contribución y retorno por canal y año.
- `outputs/sensibilidad_roi.csv`: rango de retorno en las 18 combinaciones por canal.
- `outputs/sensibilidad_seno_coseno_validacion.csv`: comparación previa a prueba de la especificación armónica frente a la simplificada.
- `outputs/hiperparametros_paso4.json`: parámetros, gammas resueltos, semilla y ventanas.
- `outputs/errores_por_segmento.csv`: error por marca y subcanal.
- `outputs/figures/`: RMSE, contribuciones y curvas de respuesta.

A5 reproduce el procedimiento individual anterior únicamente como comparación histórica; no participa en la selección del modelo final.
