"""Joint temporal calibration of all media channels with Optuna TPE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from .data import (
    MEDIA_COLS,
    MEDIA_KEYS,
    TEST_START,
    VALIDATION_END,
    VALIDATION_START,
)
from .evaluation import expanding_windows
from .models import ModelSpecification, fit_and_predict
from .transformations import (
    PARAMETER_GRID,
    adstock_by_brand,
    complete_media_panel,
    grid_parameters,
    hill_saturation,
    merge_media_features,
    training_gamma,
)


@dataclass
class CalibrationResult:
    parameters: dict[str, dict[str, float]]
    best_value: float
    history: pd.DataFrame
    windows: list[tuple[pd.Timestamp, pd.Timestamp]]
    study: optuna.Study


class CandidateFeatureCache:
    def __init__(self, media: pd.DataFrame):
        self.media = complete_media_panel(media)
        self._adstock: dict[tuple[str, float], np.ndarray] = {}
        self._feature: dict[tuple[pd.Timestamp, str, int], np.ndarray] = {}

    def feature(self, train_end: pd.Timestamp, channel: str, candidate: int) -> np.ndarray:
        key = (pd.Timestamp(train_end), channel, int(candidate))
        if key not in self._feature:
            theta, alpha, quantile = PARAMETER_GRID[int(candidate)]
            adstock_key = (channel, theta)
            if adstock_key not in self._adstock:
                self._adstock[adstock_key] = adstock_by_brand(self.media, channel, theta)
            adstock = self._adstock[adstock_key]
            gamma = training_gamma(adstock, self.media["week"], train_end, quantile)
            self._feature[key] = hill_saturation(adstock, alpha, gamma)
        return self._feature[key]

    def frame(
        self,
        train_end: pd.Timestamp,
        candidate_by_channel: dict[str, int],
    ) -> pd.DataFrame:
        output = self.media[MEDIA_KEYS].copy()
        for channel in MEDIA_COLS:
            output[f"media__{channel}"] = self.feature(
                train_end,
                channel,
                candidate_by_channel[channel],
            )
        return output


def assert_selection_excludes_test(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> None:
    latest = max(pd.to_datetime(train["week"]).max(), pd.to_datetime(evaluation["week"]).max())
    if latest >= TEST_START:
        raise AssertionError("Model selection attempted to use test observations")


def _window_score(
    panel: pd.DataFrame,
    feature_cache: CandidateFeatureCache,
    candidate_by_channel: dict[str, int],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    train = panel[panel["week"] < start].copy()
    validation = panel[panel["week"].between(start, end)].copy()
    assert_selection_excludes_test(train, validation)
    train_end = pd.Timestamp(start) - pd.Timedelta(weeks=1)
    features = feature_cache.frame(train_end, candidate_by_channel)
    train = merge_media_features(train, features)
    validation = merge_media_features(validation, features)
    specification = ModelSpecification(
        name="Paso 4 - MMM calibrado conjunto",
        include_media=True,
        constrain_media=True,
    )
    media_columns = [f"media__{channel}" for channel in MEDIA_COLS]
    _, prediction, evaluated = fit_and_predict(
        train,
        validation,
        specification,
        media_columns,
    )
    error = evaluated["volume_hl"].to_numpy(dtype=float) - prediction
    return float(np.sqrt(np.mean(np.square(error))))


def calibrate_jointly(
    panel: pd.DataFrame,
    media: pd.DataFrame,
    n_trials: int = 2000,
    seed: int = 2026,
    storage: str | None = None,
    study_name: str = "mmm_joint_calibration",
) -> CalibrationResult:
    windows = expanding_windows(panel, VALIDATION_START, VALIDATION_END, window_size=8)
    if not windows:
        raise ValueError("No complete 8-week validation windows were found")
    cache = CandidateFeatureCache(media)

    def objective(trial: optuna.Trial) -> float:
        candidates = {
            channel: trial.suggest_int(f"candidate__{channel}", 0, len(PARAMETER_GRID) - 1)
            for channel in MEDIA_COLS
        }
        scores = []
        for step, (start, end) in enumerate(windows):
            score = _window_score(panel, cache, candidates, start, end)
            scores.append(score)
            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(50, max(5, n_trials // 10)))
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage),
    )
    remaining = max(0, n_trials - len(study.trials)) if storage else n_trials
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    best_indices = {channel: int(study.best_params[f"candidate__{channel}"]) for channel in MEDIA_COLS}
    parameters = grid_parameters(best_indices)
    history = study.trials_dataframe()
    return CalibrationResult(parameters, float(study.best_value), history, windows, study)


def evaluate_parameter_set_on_validation_windows(
    panel: pd.DataFrame,
    media: pd.DataFrame,
    parameters: dict[str, dict[str, float]],
) -> tuple[float, list[float]]:
    cache = CandidateFeatureCache(media)
    parameter_to_index = {tuple(values): index for index, values in enumerate(PARAMETER_GRID)}
    indices = {
        channel: parameter_to_index[
            (
                float(parameters[channel]["theta"]),
                float(parameters[channel]["alpha"]),
                float(parameters[channel]["gamma_quantile"]),
            )
        ]
        for channel in MEDIA_COLS
    }
    windows = expanding_windows(panel, VALIDATION_START, VALIDATION_END, window_size=8)
    scores = [_window_score(panel, cache, indices, start, end) for start, end in windows]
    return float(np.mean(scores)), scores


def save_calibration(result: CalibrationResult, output: str | Path) -> None:
    import json

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": "mean expanding-window RMSE in hL",
        "best_value": result.best_value,
        "validation_windows": [
            {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}
            for start, end in result.windows
        ],
        "parameters": result.parameters,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
