import numpy as np
import pandas as pd
import pytest

from mmm.calibration import assert_selection_excludes_test
from mmm.data import (
    CALIBRATION_END,
    CALIBRATION_START,
    TEST_END,
    TEST_START,
    VALIDATION_END,
    VALIDATION_START,
    add_exact_lag52,
)
from mmm.transformations import training_gamma


def test_fixed_split_dates_are_exact_and_disjoint():
    assert CALIBRATION_START == pd.Timestamp("2023-01-07")
    assert CALIBRATION_END == pd.Timestamp("2024-11-23")
    assert VALIDATION_START == pd.Timestamp("2024-11-30")
    assert VALIDATION_END == pd.Timestamp("2025-05-17")
    assert TEST_START == pd.Timestamp("2025-05-24")
    assert TEST_END == pd.Timestamp("2025-12-27")
    assert CALIBRATION_END < VALIDATION_START <= VALIDATION_END < TEST_START <= TEST_END


def test_model_selection_rejects_test_observations():
    train = pd.DataFrame({"week": pd.to_datetime(["2023-01-07", "2024-11-23"])})
    validation = pd.DataFrame({"week": pd.to_datetime(["2024-11-30", "2025-05-17"])})
    assert_selection_excludes_test(train, validation)

    contaminated = pd.DataFrame({"week": pd.to_datetime(["2025-05-17", "2025-05-24"])})
    with pytest.raises(AssertionError, match="test observations"):
        assert_selection_excludes_test(train, contaminated)


def test_gamma_uses_training_values_only():
    weeks = pd.Series(pd.date_range("2023-01-07", periods=8, freq="7D"))
    adstock = np.array([1, 2, 3, 4, 1000, 2000, 3000, 4000], dtype=float)
    train_end = weeks.iloc[3]
    gamma = training_gamma(adstock, weeks, train_end, 0.50)
    changed_future = adstock.copy()
    changed_future[4:] = 1e12
    assert gamma == training_gamma(changed_future, weeks, train_end, 0.50)
    assert gamma == np.quantile([1, 2, 3, 4], 0.50)


def test_lag52_is_aligned_within_series():
    weeks = pd.to_datetime(["2022-01-08", "2023-01-07", "2022-01-15", "2023-01-14"])
    data = pd.DataFrame(
        {
            "week": weeks,
            "brand": ["A"] * 4,
            "region": ["R"] * 4,
            "subchannel": ["S"] * 4,
            "volume_hl": [10.0, 20.0, 30.0, 40.0],
        }
    )
    result, _ = add_exact_lag52(data)
    observed = result.set_index("week")["volume_hl_lag52"]
    assert observed.loc[pd.Timestamp("2023-01-07")] == 10.0
    assert observed.loc[pd.Timestamp("2023-01-14")] == 30.0


def test_lag52_never_crosses_series():
    data = pd.DataFrame(
        {
            "week": pd.to_datetime(["2022-01-08", "2023-01-07"]),
            "brand": ["A", "B"],
            "region": ["R", "R"],
            "subchannel": ["S", "S"],
            "volume_hl": [10.0, 20.0],
        }
    )
    result, _ = add_exact_lag52(data)
    assert result.loc[result["week"].eq(pd.Timestamp("2023-01-07")), "volume_hl_lag52"].isna().all()
