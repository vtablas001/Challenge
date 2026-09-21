"""End-to-end orchestration for the final MMM model sequence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calibration import (
    CalibrationResult,
    calibrate_jointly,
    evaluate_parameter_set_on_validation_windows,
    save_calibration,
)
from .data import (
    CALIBRATION_END,
    MEDIA_COLS,
    TEST_END,
    TEST_START,
    VALIDATION_END,
    DataBundle,
    load_model_data,
    period_masks,
)
from .evaluation import (
    error_breakdown,
    expanding_windows,
    percent_comparison,
    regression_metrics,
    seasonal_naive,
    summarize_backtest,
)
from .models import (
    REFERENCE_LEVELS,
    ModelSpecification,
    fit_and_predict,
    fit_panel_model,
)
from .transformations import (
    LEGACY_INDIVIDUAL_PARAMETERS,
    PARAMETER_GRID,
    STANDARD_PARAMETERS,
    adstock_by_brand,
    complete_media_panel,
    hill_saturation,
    legacy_individual_media_features,
    merge_media_features,
    raw_media_features,
    training_gamma,
    transform_media,
)


MEDIA_FEATURES = [f"media__{channel}" for channel in MEDIA_COLS]
LEGACY_MEDIA_FEATURES = [f"media__{channel}" for channel in LEGACY_INDIVIDUAL_PARAMETERS]


@dataclass
class PipelineResult:
    model_table: pd.DataFrame
    backtest_detail: pd.DataFrame
    error_detail: pd.DataFrame
    contributions: pd.DataFrame
    sensitivity: pd.DataFrame
    parameters: dict[str, dict[str, float]]
    recommendation: dict[str, float | str]


def _attach_transformed(
    panel: pd.DataFrame,
    media: pd.DataFrame,
    parameters: dict[str, dict[str, float]],
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    features, resolved = transform_media(media, parameters, train_end)
    return merge_media_features(panel, features), resolved


def _specifications() -> dict[str, ModelSpecification]:
    return {
        "Paso 1 - Venta base": ModelSpecification("Paso 1 - Venta base"),
        "Paso 2 - Venta base + lag52": ModelSpecification(
            "Paso 2 - Venta base + lag52", include_lag52=True
        ),
        "Paso 3 - MMM estandar": ModelSpecification(
            "Paso 3 - MMM estandar", include_media=True
        ),
        "Paso 4 - MMM calibrado conjunto": ModelSpecification(
            "Paso 4 - MMM calibrado conjunto", include_media=True
        ),
        "Paso 5 - MMM calibrado + lag52": ModelSpecification(
            "Paso 5 - MMM calibrado + lag52", include_media=True, include_lag52=True
        ),
        "A1 - Medios en nivel": ModelSpecification("A1 - Medios en nivel", include_media=True),
        "A2 - MMM calibrado + semanas": ModelSpecification(
            "A2 - MMM calibrado + semanas", include_media=True, seasonality="week_indicators"
        ),
        "A3 - MMM calibrado + semanas + lag52": ModelSpecification(
            "A3 - MMM calibrado + semanas + lag52",
            include_media=True,
            include_lag52=True,
            seasonality="week_indicators",
        ),
        "A4 - MMM calibrado + FE por serie": ModelSpecification(
            "A4 - MMM calibrado + FE por serie", include_media=True, fixed_effects="series"
        ),
        "A5 - MMM calibrado exploratorio individual": ModelSpecification(
            "A5 - MMM calibrado exploratorio individual",
            include_media=True,
            seasonality="harmonic",
            constrain_media=False,
        ),
        "A6 - MMM calibrado + smearing subcanal": ModelSpecification(
            "A6 - MMM calibrado + smearing subcanal", include_media=True
        ),
    }


def _fit_final_models(
    bundle: DataBundle,
    parameters: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, np.ndarray]], dict[str, object]]:
    panel = bundle.panel
    masks = period_masks(panel)
    development = panel.loc[masks["development"]].copy()
    test = panel.loc[masks["test"]].copy()
    calibrated, _ = _attach_transformed(panel, bundle.media, parameters, VALIDATION_END)
    standard, _ = _attach_transformed(panel, bundle.media, STANDARD_PARAMETERS, VALIDATION_END)
    raw = merge_media_features(panel, raw_media_features(bundle.media))
    legacy = merge_media_features(panel, legacy_individual_media_features(bundle.media))
    specs = _specifications()
    rows: list[dict] = []
    predictions: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    fitted: dict[str, object] = {}

    naive_prediction, naive_test = seasonal_naive(test)
    rows.append(regression_metrics("Paso 0 - Estacional ingenuo", development, naive_test, naive_prediction))
    predictions["Paso 0 - Estacional ingenuo"] = (naive_test, naive_prediction)

    datasets = {
        "Paso 1 - Venta base": panel,
        "Paso 2 - Venta base + lag52": panel,
        "Paso 3 - MMM estandar": standard,
        "Paso 4 - MMM calibrado conjunto": calibrated,
        "Paso 5 - MMM calibrado + lag52": calibrated,
        "A1 - Medios en nivel": raw,
        "A2 - MMM calibrado + semanas": calibrated,
        "A3 - MMM calibrado + semanas + lag52": calibrated,
        "A4 - MMM calibrado + FE por serie": calibrated,
        "A5 - MMM calibrado exploratorio individual": legacy,
        "A6 - MMM calibrado + smearing subcanal": calibrated,
    }

    for name, specification in specs.items():
        data = datasets[name]
        train = data.loc[data["week"].between(development["week"].min(), VALIDATION_END)].copy()
        evaluation = data.loc[data["week"].between(TEST_START, TEST_END)].copy()
        media_columns = (
            LEGACY_MEDIA_FEATURES
            if name.startswith("A5")
            else MEDIA_FEATURES if specification.include_media else []
        )
        smearing = "subchannel" if name.startswith("A6") else "global"
        model, prediction, evaluated = fit_and_predict(
            train,
            evaluation,
            specification,
            media_columns,
            smearing=smearing,
        )
        rows.append(regression_metrics(name, train, evaluated, prediction))
        predictions[name] = (evaluated, prediction)
        fitted[name] = model

    table = pd.DataFrame(rows)
    table["tipo"] = np.where(table["modelo"].str.startswith("Paso"), "principal", "apendice")
    table["paso"] = table["modelo"].str.split(" - ").str[0]
    table["estado"] = "Estimado"
    table.loc[table["modelo"].str.startswith("A5"), "estado"] = (
        "Reproduccion historica; no participa en seleccion"
    )
    return table, predictions, fitted


def _backtest(
    bundle: DataBundle,
    parameters: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = bundle.panel
    windows = expanding_windows(panel, TEST_START, TEST_END, window_size=8)
    specs = _specifications()
    model_names = list(specs)
    rows: list[dict] = []

    for fold, (start, end) in enumerate(windows, start=1):
        train_end = start - pd.Timedelta(weeks=1)
        train = panel[panel["week"] < start].copy()
        evaluation = panel[panel["week"].between(start, end)].copy()
        calibrated, _ = _attach_transformed(panel, bundle.media, parameters, train_end)
        standard, _ = _attach_transformed(panel, bundle.media, STANDARD_PARAMETERS, train_end)
        raw = merge_media_features(panel, raw_media_features(bundle.media))
        legacy = merge_media_features(panel, legacy_individual_media_features(bundle.media))

        naive_prediction, naive_eval = seasonal_naive(evaluation)
        metric = regression_metrics("Paso 0 - Estacional ingenuo", train, naive_eval, naive_prediction)
        rows.append({**metric, "fold": fold, "inicio": start, "fin": end})

        for name in model_names:
            specification = specs[name]
            if name.startswith("Paso 3"):
                data = standard
            elif name.startswith(("Paso 4", "Paso 5", "A2", "A3", "A4", "A6")):
                data = calibrated
            elif name.startswith("A1"):
                data = raw
            elif name.startswith("A5"):
                data = legacy
            else:
                data = panel
            train_fold = data[data["week"] < start].copy()
            eval_fold = data[data["week"].between(start, end)].copy()
            media_columns = (
                LEGACY_MEDIA_FEATURES
                if name.startswith("A5")
                else MEDIA_FEATURES if specification.include_media else []
            )
            _, prediction, evaluated = fit_and_predict(
                train_fold,
                eval_fold,
                specification,
                media_columns,
                smearing="subchannel" if name.startswith("A6") else "global",
            )
            metric = regression_metrics(name, train_fold, evaluated, prediction)
            rows.append({**metric, "fold": fold, "inicio": start, "fin": end})

    detail = pd.DataFrame(rows)
    summary = summarize_backtest(detail.to_dict("records"))
    return detail, summary


def _key_comparisons(model_table: pd.DataFrame) -> pd.DataFrame:
    indexed = model_table.set_index("paso")
    pairs = [("Paso 3", "Paso 1"), ("Paso 4", "Paso 1"), ("Paso 4", "Paso 3"), ("Paso 5", "Paso 2"), ("Paso 5", "Paso 0")]
    rows = []
    for model, reference in pairs:
        a, b = indexed.loc[model], indexed.loc[reference]
        rows.append(
            {
                "comparacion": f"{model} vs {reference}",
                "mejora_RMSE_pct": percent_comparison(a, b, "RMSE_hl"),
                "mejora_MAE_pct": percent_comparison(a, b, "MAE_hl"),
            }
        )
    return pd.DataFrame(rows)


def _attribution(
    bundle: DataBundle,
    parameters: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, object, pd.DataFrame, dict[str, dict[str, float]]]:
    panel, resolved = _attach_transformed(bundle.panel, bundle.media, parameters, TEST_END)
    specification = ModelSpecification("Paso 4 - MMM calibrado conjunto", include_media=True)
    model = fit_panel_model(panel, specification, MEDIA_FEATURES)
    observed = model.predict(panel)
    rows = []
    media = bundle.media.copy()
    media["year"] = media["week"].dt.year

    for channel in MEDIA_COLS:
        counterfactual = panel.copy()
        counterfactual[f"media__{channel}"] = 0.0
        contribution = np.maximum(observed - model.predict(counterfactual), 0.0)
        local = pd.DataFrame({"year": panel["week"].dt.year, "incremental_hl": contribution})
        by_year = local.groupby("year", as_index=False)["incremental_hl"].sum()
        spend = media.groupby("year", as_index=False)[channel].sum().rename(columns={channel: "spend"})
        by_year = by_year.merge(spend, on="year", how="left")
        by_year["channel"] = channel
        rows.extend(by_year.to_dict("records"))
        rows.append(
            {
                "year": "Total",
                "incremental_hl": float(contribution.sum()),
                "spend": float(media[channel].sum()),
                "channel": channel,
            }
        )

    result = pd.DataFrame(rows)
    revenue_lookup = bundle.revenue_per_hl.set_index("year")["revenue_per_hl"].to_dict()
    total_revenue_per_hl = float(
        bundle.revenue_per_hl["revenue"].sum() / bundle.revenue_per_hl["volume_hl"].sum()
    )
    result["revenue_per_hl"] = result["year"].map(revenue_lookup).fillna(total_revenue_per_hl)
    result["incremental_hl_per_spend"] = np.divide(
        result["incremental_hl"], result["spend"], out=np.full(len(result), np.nan), where=result["spend"].ne(0)
    )
    result["revenue_roi"] = np.divide(
        result["incremental_hl"] * result["revenue_per_hl"],
        result["spend"],
        out=np.full(len(result), np.nan),
        where=result["spend"].ne(0),
    )
    return result, model, panel, resolved


def _sensitivity(
    bundle: DataBundle,
    selected_parameters: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    total_spend = bundle.media[MEDIA_COLS].sum()
    for channel in MEDIA_COLS:
        returns = []
        for theta, alpha, quantile in PARAMETER_GRID:
            parameters = {name: values.copy() for name, values in selected_parameters.items()}
            parameters[channel] = {
                "theta": theta,
                "alpha": alpha,
                "gamma_quantile": quantile,
            }
            data, _ = _attach_transformed(bundle.panel, bundle.media, parameters, TEST_END)
            model = fit_panel_model(
                data,
                ModelSpecification("Sensitivity", include_media=True),
                MEDIA_FEATURES,
            )
            observed = model.predict(data)
            zero = data.copy()
            zero[f"media__{channel}"] = 0.0
            incremental = float(np.maximum(observed - model.predict(zero), 0.0).sum())
            returns.append(incremental / total_spend[channel] if total_spend[channel] else np.nan)
        valid = np.asarray(returns, dtype=float)
        rows.append(
            {
                "channel": channel,
                "return_min_hl_per_spend": float(np.nanmin(valid)),
                "return_median_hl_per_spend": float(np.nanmedian(valid)),
                "return_max_hl_per_spend": float(np.nanmax(valid)),
                "combinations": len(PARAMETER_GRID),
            }
        )
    return pd.DataFrame(rows)


def _save_figures(
    output_dir: Path,
    model_table: pd.DataFrame,
    contributions: pd.DataFrame,
    bundle: DataBundle,
    resolved: dict[str, dict[str, float]],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    estimated = model_table[model_table["RMSE_hl"].notna()].sort_values("RMSE_hl")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(estimated["modelo"], estimated["RMSE_hl"], color="#176B87")
    ax.invert_yaxis()
    ax.set_xlabel("RMSE de prueba (hL)")
    ax.set_title("Error fuera de muestra por modelo")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(figures / "rmse_por_modelo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    total = contributions[contributions["year"].eq("Total")].sort_values("incremental_hl")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(total["channel"], total["incremental_hl"], color="#4C956C")
    ax.set_xlabel("Volumen incremental estimado (hL)")
    ax.set_title("Contribucion estimada por canal")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(figures / "contribuciones_por_canal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    media = complete_media_panel(bundle.media)
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    for ax, channel in zip(axes.flat, MEDIA_COLS):
        params = resolved[channel]
        adstock = adstock_by_brand(media, channel, params["theta"])
        maximum = max(float(np.max(adstock)), params["gamma"] * 2, 1.0)
        x = np.linspace(0, maximum, 200)
        y = hill_saturation(x, params["alpha"], params["gamma"])
        mean = float(np.mean(adstock))
        ax.plot(x, y, color="#176B87", linewidth=2)
        ax.axvline(mean, color="#C65D3B", linestyle="--", linewidth=1)
        ax.set_title(channel, fontsize=10)
        ax.grid(linestyle=":", alpha=0.25)
    for ax in axes.flat[len(MEDIA_COLS) :]:
        ax.axis("off")
    fig.suptitle("Curvas de respuesta Hill; linea punteada = adstock medio", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures / "curvas_respuesta_hill.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(
    root: str | Path,
    n_trials: int = 2000,
    seed: int = 2026,
    run_sensitivity: bool = True,
) -> PipelineResult:
    root = Path(root)
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    optuna_storage = f"sqlite:///{(output_dir / 'optuna_mmm.db').as_posix()}"

    bundle = load_model_data(root)
    calibration = calibrate_jointly(
        bundle.panel,
        bundle.media,
        n_trials=n_trials,
        seed=seed,
        storage=optuna_storage,
        study_name="mmm_joint_calibration_no_harmonics",
    )
    parameter_path = output_dir / "hiperparametros_paso4.json"
    save_calibration(calibration, parameter_path)
    _, resolved_development = transform_media(
        bundle.media,
        calibration.parameters,
        VALIDATION_END,
    )
    parameter_payload = json.loads(parameter_path.read_text(encoding="utf-8"))
    parameter_payload.update(
        {
            "seed": seed,
            "requested_trials": n_trials,
            "recorded_trials": len(calibration.study.trials),
            "resolved_for_development_fit": resolved_development,
            "test_used_for_selection": False,
        }
    )
    parameter_path.write_text(
        json.dumps(parameter_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    calibration.history.to_csv(output_dir / "historial_optuna.csv", index=False)

    standard_rmse, standard_windows = evaluate_parameter_set_on_validation_windows(
        bundle.panel, bundle.media, STANDARD_PARAMETERS
    )
    calibrated_rmse = calibration.best_value
    standard_loss = 100 * (standard_rmse - calibrated_rmse) / calibrated_rmse
    recommendation = {
        "rmse_validacion_standard": standard_rmse,
        "rmse_validacion_calibrated": calibrated_rmse,
        "standard_loss_pct": standard_loss,
        "selected_model": "Paso 3 - MMM estandar" if standard_loss <= 5 else "Paso 4 - MMM calibrado conjunto",
        "rule": "Prefer standard when its validation RMSE is at most 5% worse",
        "reference_levels": {**REFERENCE_LEVELS, "media_business_comparator": "localtv"},
    }
    model_table, predictions, _ = _fit_final_models(bundle, calibration.parameters)
    backtest_detail, backtest_summary = _backtest(bundle, calibration.parameters)
    model_table = model_table.merge(backtest_summary, on="modelo", how="left")
    comparisons = _key_comparisons(model_table)
    selected_predictions = {
        key: predictions[key]
        for key in [
            "Paso 0 - Estacional ingenuo",
            "Paso 1 - Venta base",
            "Paso 4 - MMM calibrado conjunto",
            "Paso 5 - MMM calibrado + lag52",
        ]
    }
    error_detail = error_breakdown(selected_predictions)
    contributions, _, _, resolved = _attribution(bundle, calibration.parameters)
    localtv_reference = (
        contributions.loc[
            contributions["channel"].eq("localtv"),
            ["year", "incremental_hl_per_spend", "revenue_roi"],
        ]
        .set_index("year")
    )
    contributions["efficiency_index_vs_localtv"] = contributions.apply(
        lambda row: row["incremental_hl_per_spend"]
        / localtv_reference.at[row["year"], "incremental_hl_per_spend"],
        axis=1,
    )
    contributions["roi_index_vs_localtv"] = contributions.apply(
        lambda row: row["revenue_roi"] / localtv_reference.at[row["year"], "revenue_roi"],
        axis=1,
    )
    sensitivity = _sensitivity(bundle, calibration.parameters) if run_sensitivity else pd.DataFrame()

    naive_rmse = float(model_table.loc[model_table["paso"].eq("Paso 0"), "RMSE_hl"].iloc[0])
    series_rmse = float(model_table.loc[model_table["paso"].eq("A4"), "RMSE_hl"].iloc[0])
    base_gap = float(model_table.loc[model_table["paso"].eq("Paso 4"), "RMSE_hl"].iloc[0]) - naive_rmse
    series_gap = series_rmse - naive_rmse
    recommendation["paso4_rmse_gap_vs_naive_hl"] = base_gap
    recommendation["A4_rmse_gap_vs_naive_hl"] = series_gap
    recommendation["A4_gap_reduction_vs_naive_pct"] = 100 * (base_gap - series_gap) / base_gap if base_gap else np.nan
    recommendation["lag52_affected_series"] = int(len(bundle.lag52_issues))
    (output_dir / "decision_modelo.json").write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    model_table.to_csv(output_dir / "tabla_modelos.csv", index=False)
    backtest_detail.to_csv(output_dir / "backtesting_detalle.csv", index=False)
    error_detail.to_csv(output_dir / "errores_por_segmento.csv", index=False)
    comparisons.to_csv(output_dir / "comparaciones_clave.csv", index=False)
    contributions.to_csv(output_dir / "contribuciones_roi.csv", index=False)
    sensitivity.to_csv(output_dir / "sensibilidad_roi.csv", index=False)
    bundle.lag52_issues.to_csv(output_dir / "series_con_lag52_faltante.csv", index=False)
    _save_figures(output_dir, model_table, contributions, bundle, resolved)
    return PipelineResult(
        model_table=model_table,
        backtest_detail=backtest_detail,
        error_detail=error_detail,
        contributions=contributions,
        sensitivity=sensitivity,
        parameters=calibration.parameters,
        recommendation=recommendation,
    )
