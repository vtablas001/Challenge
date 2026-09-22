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


def test_supported_seasonality_is_explicit():
    numeric, categorical = control_columns(ModelSpecification("main"))

    assert "trend" not in numeric
    assert categorical == ["brand", "region", "subchannel"]

    with np.testing.assert_raises_regex(ValueError, "Unknown seasonality"):
        control_columns(ModelSpecification("unsupported", seasonality="cyclical"))


def test_brand_and_subchannel_reference_levels_are_explicit():
    data = pd.DataFrame(
        {
            "brand": ["Brand A", "Brand B", "Brand C"],
            "region": ["Region A", "Region B", "Region C"],
            "subchannel": ["Subchannel A", "Subchannel B", "Subchannel C"],
        }
    )
    from mmm.models import DesignEncoder

    encoder = DesignEncoder(
        numeric=[],
        categorical=["brand", "region", "subchannel"],
    ).fit(data)

    assert encoder.levels["brand"][0] == "Brand C"
    assert encoder.levels["subchannel"][0] == "Subchannel B"
    assert "brand[Brand C]" not in encoder.feature_names
    assert "subchannel[Subchannel B]" not in encoder.feature_names
    assert encoder.levels["region"][0] == "Region A"
