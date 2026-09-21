"""Temporal evaluation, benchmark metrics and reporting helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import SERIES_COLS


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    return float(np.abs(actual - predicted).sum() / denominator) if denominator else np.nan


def seasonal_scales(train: pd.DataFrame) -> pd.Series:
    lookup = train[["week", *SERIES_COLS, "volume_hl"]].copy()
    lag = lookup.copy()
    lag["week"] = lag["week"] + pd.Timedelta(weeks=52)
    lag = lag.rename(columns={"volume_hl": "lag52"})
    paired = lookup.merge(lag, on=["week", *SERIES_COLS], how="left", validate="one_to_one")
    paired["absolute_seasonal_error"] = (paired["volume_hl"] - paired["lag52"]).abs()
    paired["series_id"] = paired[SERIES_COLS].astype(str).agg(" | ".join, axis=1)
    return paired.groupby("series_id")["absolute_seasonal_error"].mean().dropna()


def mase_values(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    predicted: np.ndarray,
) -> tuple[float, float]:
    scales = seasonal_scales(train)
    frame = evaluation[[*SERIES_COLS, "volume_hl"]].copy()
    frame["series_id"] = frame[SERIES_COLS].astype(str).agg(" | ".join, axis=1)
    frame["absolute_error"] = np.abs(frame["volume_hl"].to_numpy(dtype=float) - predicted)
    frame["scale"] = frame["series_id"].map(scales)
    valid = frame["scale"].gt(0) & frame["scale"].notna()
    per_series = (
        frame.loc[valid]
        .groupby("series_id")
        .agg(mae=("absolute_error", "mean"), scale=("scale", "first"))
    )
    mase_average = float((per_series["mae"] / per_series["scale"]).mean())
    mase_aggregate = float(frame.loc[valid, "absolute_error"].sum() / frame.loc[valid, "scale"].sum())
    return mase_average, mase_aggregate


def regression_metrics(
    name: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    predicted: np.ndarray,
) -> dict[str, float | int | str]:
    actual = evaluation["volume_hl"].to_numpy(dtype=float)
    mase_average, mase_aggregate = mase_values(train, evaluation, predicted)
    return {
        "modelo": name,
        "n_obs": int(len(actual)),
        "R2": float(r2_score(actual, predicted)),
        "RMSE_hl": float(mean_squared_error(actual, predicted) ** 0.5),
        "MAE_hl": float(mean_absolute_error(actual, predicted)),
        "WAPE": wape(actual, predicted),
        "MASE_promedio_series": mase_average,
        "MASE_agregado": mase_aggregate,
    }


def seasonal_naive(evaluation: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    mask = evaluation["volume_hl_lag52"].notna()
    frame = evaluation.loc[mask].copy()
    return frame["volume_hl_lag52"].to_numpy(dtype=float), frame


def expanding_windows(
    data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_size: int = 8,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    weeks = np.array(sorted(data.loc[data["week"].between(start, end), "week"].unique()))
    windows = []
    for offset in range(0, len(weeks), window_size):
        window = weeks[offset : offset + window_size]
        if len(window) == window_size:
            windows.append((pd.Timestamp(window[0]), pd.Timestamp(window[-1])))
    return windows


def summarize_backtest(rows: list[dict[str, float | str | int]]) -> pd.DataFrame:
    detail = pd.DataFrame(rows)
    return (
        detail.groupby("modelo", as_index=False)
        .agg(
            ventanas=("fold", "nunique"),
            RMSE_medio=("RMSE_hl", "mean"),
            RMSE_std=("RMSE_hl", "std"),
            MAE_medio=("MAE_hl", "mean"),
            MAE_std=("MAE_hl", "std"),
            WAPE_medio=("WAPE", "mean"),
            WAPE_std=("WAPE", "std"),
        )
        .sort_values("RMSE_medio")
        .reset_index(drop=True)
    )


def error_breakdown(
    predictions: Mapping[str, tuple[pd.DataFrame, np.ndarray]],
    dimensions: tuple[str, ...] = ("brand", "subchannel"),
) -> pd.DataFrame:
    rows = []
    for model_name, (frame, predicted) in predictions.items():
        local = frame.copy()
        local["prediction"] = predicted
        local["absolute_error"] = (local["volume_hl"] - local["prediction"]).abs()
        for dimension in dimensions:
            for value, group in local.groupby(dimension):
                rows.append(
                    {
                        "modelo": model_name,
                        "dimension": dimension,
                        "segmento": value,
                        "n_obs": len(group),
                        "MAE_hl": group["absolute_error"].mean(),
                        "WAPE": group["absolute_error"].sum() / group["volume_hl"].abs().sum(),
                    }
                )
    return pd.DataFrame(rows)


def percent_comparison(a: pd.Series, b: pd.Series, metric: str) -> float:
    """Improvement of model a relative to model b; positive is better."""
    return float(100 * (b[metric] - a[metric]) / b[metric])
