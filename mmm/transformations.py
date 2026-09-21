"""Media transformations with explicit, training-only calibration rules."""

from __future__ import annotations

from itertools import product
from typing import Mapping

import numpy as np
import pandas as pd

from .data import MEDIA_COLS, MEDIA_KEYS


THETA_GRID = (0.3, 0.5, 0.7)
ALPHA_GRID = (1.0, 2.0)
GAMMA_QUANTILES = (0.25, 0.50, 0.75)
PARAMETER_GRID = tuple(product(THETA_GRID, ALPHA_GRID, GAMMA_QUANTILES))
STANDARD_PARAMETERS = {
    channel: {"theta": 0.5, "alpha": 1.0, "gamma_quantile": 0.50}
    for channel in MEDIA_COLS
}

def geometric_adstock(values: np.ndarray | pd.Series, theta: float) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    result = np.zeros_like(values, dtype=float)
    for index, value in enumerate(values):
        result[index] = value + (theta * result[index - 1] if index else 0.0)
    return result


def hill_saturation(values: np.ndarray | pd.Series, alpha: float, gamma: float) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0, None)
    gamma = max(float(gamma), np.finfo(float).eps)
    numerator = np.power(values, float(alpha))
    denominator = numerator + gamma ** float(alpha)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def complete_media_panel(media: pd.DataFrame) -> pd.DataFrame:
    """Insert missing brand-weeks as zero spend so adstock decays every week."""
    weeks = pd.date_range(media["week"].min(), media["week"].max(), freq="7D")
    brands = sorted(media["brand"].unique())
    index = pd.MultiIndex.from_product([weeks, brands], names=MEDIA_KEYS)
    completed = media.set_index(MEDIA_KEYS).reindex(index).reset_index()
    completed[MEDIA_COLS] = completed[MEDIA_COLS].fillna(0.0)
    return completed.sort_values(["brand", "week"]).reset_index(drop=True)


def adstock_by_brand(media: pd.DataFrame, channel: str, theta: float) -> np.ndarray:
    output = pd.Series(index=media.index, dtype=float)
    for _, indices in media.groupby("brand", sort=False).groups.items():
        output.loc[indices] = geometric_adstock(media.loc[indices, channel], theta)
    return output.to_numpy()


def training_gamma(
    adstock: np.ndarray,
    weeks: pd.Series,
    train_end: pd.Timestamp,
    quantile: float,
) -> float:
    """Calculate half-saturation solely from positive training adstock."""
    values = np.asarray(adstock, dtype=float)[pd.to_datetime(weeks).le(train_end).to_numpy()]
    values = values[values > 0]
    return float(np.quantile(values, quantile)) if len(values) else float(np.finfo(float).eps)


def transform_media(
    media: pd.DataFrame,
    parameters: Mapping[str, Mapping[str, float]],
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Apply full-history adstock and training-only Hill calibration."""
    completed = complete_media_panel(media)
    transformed = completed[MEDIA_KEYS].copy()
    resolved: dict[str, dict[str, float]] = {}

    for channel in MEDIA_COLS:
        params = parameters[channel]
        theta = float(params["theta"])
        alpha = float(params["alpha"])
        quantile = float(params["gamma_quantile"])
        adstock = adstock_by_brand(completed, channel, theta)
        gamma = training_gamma(adstock, completed["week"], train_end, quantile)
        column = f"media__{channel}"
        transformed[column] = hill_saturation(adstock, alpha, gamma)
        resolved[channel] = {
            "theta": theta,
            "alpha": alpha,
            "gamma_quantile": quantile,
            "gamma": gamma,
        }
    return transformed, resolved


def raw_media_features(media: pd.DataFrame) -> pd.DataFrame:
    completed = complete_media_panel(media)
    output = completed[MEDIA_KEYS].copy()
    for channel in MEDIA_COLS:
        output[f"media__{channel}"] = completed[channel].astype(float)
    return output


def merge_media_features(panel: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in features if column.startswith("media__")]
    merged = panel.merge(
        features[MEDIA_KEYS + columns],
        on=MEDIA_KEYS,
        how="left",
        validate="many_to_one",
    )
    merged[columns] = merged[columns].fillna(0.0)
    return merged


def grid_parameters(index_by_channel: Mapping[str, int]) -> dict[str, dict[str, float]]:
    result = {}
    for channel in MEDIA_COLS:
        theta, alpha, quantile = PARAMETER_GRID[int(index_by_channel[channel])]
        result[channel] = {
            "theta": theta,
            "alpha": alpha,
            "gamma_quantile": quantile,
        }
    return result
