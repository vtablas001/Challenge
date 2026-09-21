import numpy as np
import pandas as pd

from mmm.models import ModelSpecification, control_columns, fit_panel_model


def test_media_coefficients_are_bounded_not_selected_by_sign():
    n = 40
    media = np.linspace(0, 1, n)
    log_target = 5.0 - 2.0 * media
    data = pd.DataFrame(
        {
            "volume_hl": np.expm1(log_target),
            "log1p_volume_hl": log_target,
            "weighted_distribution": 1.0,
            "avg_temp": 20.0,
            "avg_prcp": 0.0,
            "evento_deportivo": 0.0,
            "evento_social": 0.0,
            "feriado": 0.0,
            "trend": 0.0,
            "week_sin": 0.0,
            "week_cos": 1.0,
            "brand": "Marca A",
            "region": "Region A",
            "subchannel": "Subcanal A",
            "media__test": media,
        }
    )
    specification = ModelSpecification(
        "constraint audit",
        include_media=True,
        constrain_media=True,
    )
    model = fit_panel_model(data, specification, ["media__test"])

    assert "media__test" in model.coefficients.index
    assert model.coefficients["media__test"] >= 0
    assert np.isclose(model.coefficients["media__test"], 0.0)


def test_main_specification_excludes_annual_harmonics_by_default():
    numeric, categorical = control_columns(ModelSpecification("main"))

    assert "week_sin" not in numeric
    assert "week_cos" not in numeric
    assert "trend" in numeric
    assert categorical == ["brand", "region", "subchannel"]
