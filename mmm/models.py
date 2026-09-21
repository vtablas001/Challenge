"""Linear panel models on log1p(volume_hl), with constrained media effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .data import BASE_NUMERIC_COLS, EVENT_COLS


@dataclass(frozen=True)
class ModelSpecification:
    name: str
    include_media: bool = False
    include_lag52: bool = False
    seasonality: str = "harmonic"
    fixed_effects: str = "additive"
    constrain_media: bool = True


class DesignEncoder:
    def __init__(self, numeric: Iterable[str], categorical: Iterable[str]):
        self.numeric = list(dict.fromkeys(numeric))
        self.categorical = list(dict.fromkeys(categorical))
        self.medians: pd.Series | None = None
        self.means: pd.Series | None = None
        self.scales: pd.Series | None = None
        self.levels: dict[str, list[str]] = {}
        self.feature_names: list[str] = []

    def fit(self, data: pd.DataFrame) -> "DesignEncoder":
        numeric = data[self.numeric].apply(pd.to_numeric, errors="coerce")
        self.medians = numeric.median().fillna(0.0)
        filled = numeric.fillna(self.medians)
        self.means = filled.mean()
        self.scales = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
        self.levels = {
            column: sorted(data[column].dropna().astype(str).unique().tolist())
            for column in self.categorical
        }
        dummy_names = [
            f"{column}[{level}]"
            for column in self.categorical
            for level in self.levels[column][1:]
        ]
        self.feature_names = ["const", *self.numeric, *dummy_names]
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("DesignEncoder must be fitted before transform")
        numeric = data[self.numeric].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(self.medians)
        numeric_values = ((numeric - self.means) / self.scales).to_numpy(dtype=float)
        parts = [np.ones((len(data), 1)), numeric_values]
        for column in self.categorical:
            values = data[column].astype(str)
            for level in self.levels[column][1:]:
                parts.append(values.eq(level).to_numpy(dtype=float)[:, None])
        return np.column_stack(parts)

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.fit(data).transform(data)


@dataclass
class PanelLogModel:
    specification: ModelSpecification
    encoder: DesignEncoder
    control_coefficients: np.ndarray
    media_columns: list[str]
    media_means: pd.Series
    media_scales: pd.Series
    media_coefficients: np.ndarray
    smearing_global: float
    smearing_subchannel: dict[str, float]

    @property
    def coefficients(self) -> pd.Series:
        names = self.encoder.feature_names + self.media_columns
        values = np.concatenate([self.control_coefficients, self.media_coefficients])
        return pd.Series(values, index=names, name="coefficient")

    def _media_matrix(self, data: pd.DataFrame) -> np.ndarray:
        if not self.media_columns:
            return np.empty((len(data), 0))
        raw = data[self.media_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return ((raw - self.media_means) / self.media_scales).to_numpy(dtype=float)

    def predict_log(self, data: pd.DataFrame) -> np.ndarray:
        controls = self.encoder.transform(data)
        prediction = controls @ self.control_coefficients
        if self.media_columns:
            prediction = prediction + self._media_matrix(data) @ self.media_coefficients
        return np.asarray(prediction, dtype=float)

    def predict(self, data: pd.DataFrame, smearing: str = "global") -> np.ndarray:
        prediction_log = self.predict_log(data)
        if smearing == "subchannel":
            factors = (
                data["subchannel"].map(self.smearing_subchannel).fillna(self.smearing_global).to_numpy()
            )
        elif smearing == "global":
            factors = self.smearing_global
        else:
            raise ValueError("smearing must be 'global' or 'subchannel'")
        return np.maximum(np.exp(prediction_log) * factors - 1.0, 0.0)


def control_columns(specification: ModelSpecification) -> tuple[list[str], list[str]]:
    numeric = [*BASE_NUMERIC_COLS, *EVENT_COLS, "trend"]
    categorical: list[str]
    if specification.seasonality == "harmonic":
        numeric.extend(["week_sin", "week_cos"])
    elif specification.seasonality == "week_indicators":
        pass
    else:
        raise ValueError(f"Unknown seasonality: {specification.seasonality}")

    if specification.include_lag52:
        numeric.append("log1p_volume_hl_lag52")

    if specification.fixed_effects == "additive":
        categorical = ["brand", "region", "subchannel"]
    elif specification.fixed_effects == "series":
        categorical = ["series_id"]
    else:
        raise ValueError(f"Unknown fixed effects: {specification.fixed_effects}")

    if specification.seasonality == "week_indicators":
        categorical.append("week_of_year")
    return numeric, categorical


def eligible_rows(data: pd.DataFrame, specification: ModelSpecification) -> pd.Series:
    mask = data["volume_hl"].notna()
    if specification.include_lag52:
        mask &= data["log1p_volume_hl_lag52"].notna()
    return mask


def _fit_coefficients(
    controls: np.ndarray,
    media: np.ndarray,
    target: np.ndarray,
    constrain_media: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if media.shape[1] == 0:
        return np.linalg.lstsq(controls, target, rcond=None)[0], np.empty(0)
    if not constrain_media:
        coefficients = np.linalg.lstsq(np.column_stack([controls, media]), target, rcond=None)[0]
        return coefficients[: controls.shape[1]], coefficients[controls.shape[1] :]

    # Frisch-Waugh-Lovell residualization leaves a small nonnegative problem.
    control_projection = np.linalg.lstsq(controls, np.column_stack([target, media]), rcond=None)[0]
    residualized = np.column_stack([target, media]) - controls @ control_projection
    media_coefficients = nnls(residualized[:, 1:], residualized[:, 0])[0]
    control_coefficients = np.linalg.lstsq(
        controls,
        target - media @ media_coefficients,
        rcond=None,
    )[0]
    return control_coefficients, media_coefficients


def fit_panel_model(
    train: pd.DataFrame,
    specification: ModelSpecification,
    media_columns: list[str] | None = None,
) -> PanelLogModel:
    media_columns = list(media_columns or []) if specification.include_media else []
    row_mask = eligible_rows(train, specification)
    train = train.loc[row_mask].copy()
    if train.empty:
        raise ValueError(f"No eligible rows for {specification.name}")

    numeric, categorical = control_columns(specification)
    encoder = DesignEncoder(numeric, categorical)
    controls = encoder.fit_transform(train)
    target = train["log1p_volume_hl"].to_numpy(dtype=float)

    if media_columns:
        raw_media = train[media_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        media_means = raw_media.mean()
        media_scales = raw_media.std(ddof=0).replace(0, 1.0).fillna(1.0)
        media = ((raw_media - media_means) / media_scales).to_numpy(dtype=float)
    else:
        media_means = pd.Series(dtype=float)
        media_scales = pd.Series(dtype=float)
        media = np.empty((len(train), 0))

    control_coefs, media_coefs = _fit_coefficients(
        controls,
        media,
        target,
        specification.constrain_media,
    )
    fitted_log = controls @ control_coefs
    if media_columns:
        fitted_log += media @ media_coefs
    residual = target - fitted_log
    global_smearing = float(np.mean(np.exp(residual)))
    residual_frame = pd.DataFrame(
        {"subchannel": train["subchannel"].to_numpy(), "exp_residual": np.exp(residual)}
    )
    subchannel_smearing = residual_frame.groupby("subchannel")["exp_residual"].mean().to_dict()

    return PanelLogModel(
        specification=specification,
        encoder=encoder,
        control_coefficients=control_coefs,
        media_columns=media_columns,
        media_means=media_means,
        media_scales=media_scales,
        media_coefficients=media_coefs,
        smearing_global=global_smearing,
        smearing_subchannel=subchannel_smearing,
    )


def fit_and_predict(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    specification: ModelSpecification,
    media_columns: list[str] | None = None,
    smearing: str = "global",
) -> tuple[PanelLogModel, np.ndarray, pd.DataFrame]:
    model = fit_panel_model(train, specification, media_columns)
    evaluation_mask = eligible_rows(evaluation, specification)
    evaluated = evaluation.loc[evaluation_mask].copy()
    return model, model.predict(evaluated, smearing=smearing), evaluated
