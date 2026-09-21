"""Carga, validacion e integracion de las cuatro fuentes del challenge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CALIBRATION_START = pd.Timestamp("2023-01-07")
CALIBRATION_END = pd.Timestamp("2024-11-23")
VALIDATION_START = pd.Timestamp("2024-11-30")
VALIDATION_END = pd.Timestamp("2025-05-17")
TEST_START = pd.Timestamp("2025-05-24")
TEST_END = pd.Timestamp("2025-12-27")

SERIES_COLS = ["brand", "region", "subchannel"]
PANEL_KEYS = ["week", *SERIES_COLS]
MEDIA_KEYS = ["week", "brand"]
MEDIA_COLS = [
    "cinema",
    "digitaldisplayandsearch",
    "digitalvideo",
    "facebook",
    "instagram",
    "localtv",
    "ooh",
    "opentv",
    "paytv",
    "print",
    "radio",
    "tiktok",
    "twitter",
    "youtube",
]
BASE_NUMERIC_COLS = ["weighted_distribution", "avg_temp", "avg_prcp"]
EVENT_COLS = ["evento_deportivo", "evento_social", "feriado"]


@dataclass(frozen=True)
class DataBundle:
    panel: pd.DataFrame
    media: pd.DataFrame
    lag52_issues: pd.DataFrame
    revenue_per_hl: pd.DataFrame


def _read_csv(root: Path, number: int) -> pd.DataFrame:
    path = root / f"Challenge - Table {number}.csv"
    data = pd.read_csv(path)
    if "week" in data:
        data["week"] = pd.to_datetime(data["week"], errors="raise")
    return data


def load_tables(root: str | Path) -> dict[int, pd.DataFrame]:
    root = Path(root)
    return {number: _read_csv(root, number) for number in range(1, 5)}


def clean_context(table4: pd.DataFrame) -> pd.DataFrame:
    """Treat events and income at their native week-region granularity."""
    data = table4.copy().sort_values(["region", "week"])
    for column in ["evento_deportivo", "evento_social"]:
        observed = set(pd.to_numeric(data[column], errors="coerce").dropna().unique())
        if not observed.issubset({0, 1}):
            raise ValueError(f"{column} is not binary: {sorted(observed)}")

    holiday = pd.to_numeric(data["feriado"], errors="coerce").dropna()
    if (holiday.lt(0).any() or not np.allclose(holiday, np.round(holiday))):
        raise ValueError("feriado must be a nonnegative integer count")

    data[EVENT_COLS] = data[EVENT_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    data["income_region"] = (
        pd.to_numeric(data["income_region"], errors="coerce")
        .groupby(data["region"], sort=False)
        .ffill()
    )
    return data.sort_values(["week", "region"]).reset_index(drop=True)


def add_exact_lag52(table1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the observed value from exactly 52 weeks earlier in the same series."""
    if table1.duplicated(PANEL_KEYS).any():
        raise ValueError("Table 1 contains duplicate panel keys")

    lag_lookup = table1[PANEL_KEYS + ["volume_hl"]].copy()
    lag_lookup["week"] = lag_lookup["week"] + pd.Timedelta(weeks=52)
    lag_lookup = lag_lookup.rename(columns={"volume_hl": "volume_hl_lag52"})
    data = table1.merge(lag_lookup, on=PANEL_KEYS, how="left", validate="one_to_one")
    data["log1p_volume_hl_lag52"] = np.log1p(data["volume_hl_lag52"])

    model_period = data["week"].between(CALIBRATION_START, TEST_END)
    issues = (
        data.loc[model_period & data["volume_hl_lag52"].isna(), SERIES_COLS]
        .drop_duplicates()
        .sort_values(SERIES_COLS)
        .reset_index(drop=True)
    )
    return data, issues


def _prepare_media(table2: pd.DataFrame) -> pd.DataFrame:
    media = table2[MEDIA_KEYS + MEDIA_COLS].copy()
    if media.duplicated(MEDIA_KEYS).any():
        raise ValueError("Table 2 contains duplicate week-brand keys")
    media[MEDIA_COLS] = media[MEDIA_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    return media.sort_values(MEDIA_KEYS).reset_index(drop=True)


def integrate_tables(tables: dict[int, pd.DataFrame]) -> DataBundle:
    table1 = tables[1].copy()
    table2 = _prepare_media(tables[2])
    table3 = tables[3].copy()
    table4 = clean_context(tables[4])

    table1, lag_issues = add_exact_lag52(table1)
    distribution = table3.pivot_table(
        index=PANEL_KEYS,
        columns="variable",
        values="value",
        aggfunc="mean",
    ).reset_index()
    distribution.columns.name = None

    original_rows = len(table1)
    original_volume = float(table1["volume_hl"].sum())
    master = table1.merge(distribution, on=PANEL_KEYS, how="left", validate="one_to_one")
    master = master.merge(table4, on=["week", "region"], how="left", validate="many_to_one")
    if len(master) != original_rows or not np.isclose(master["volume_hl"].sum(), original_volume):
        raise AssertionError("Integration changed Table 1 rows or total volume")

    # Match the notebook's model universe: only week-brand keys observed in media.
    master = master.merge(table2, on=MEDIA_KEYS, how="inner", validate="many_to_one")
    master = master[master["week"].between(CALIBRATION_START, TEST_END)].copy()
    master = master.sort_values(PANEL_KEYS).reset_index(drop=True)

    master["trend"] = ((master["week"] - CALIBRATION_START).dt.days // 7).astype(float)
    master["week_of_year"] = master["week"].dt.isocalendar().week.astype(str)
    master["series_id"] = master[SERIES_COLS].astype(str).agg(" | ".join, axis=1)
    master["log1p_volume_hl"] = np.log1p(master["volume_hl"].astype(float))

    for column in ["evento_deportivo", "evento_social"]:
        values = set(master[column].dropna().unique())
        if not values.issubset({0, 1}):
            raise AssertionError(f"{column} changed coding after integration")

    revenue = table1[table1["week"].between(CALIBRATION_START, TEST_END)].copy()
    revenue["year"] = revenue["week"].dt.year
    revenue_per_hl = (
        revenue.groupby("year", as_index=False)
        .agg(revenue=("net_revenue_adj_CPI", "sum"), volume_hl=("volume_hl", "sum"))
    )
    revenue_per_hl["revenue_per_hl"] = np.divide(
        revenue_per_hl["revenue"],
        revenue_per_hl["volume_hl"],
        out=np.full(len(revenue_per_hl), np.nan),
        where=revenue_per_hl["volume_hl"].ne(0),
    )

    return DataBundle(master, table2, lag_issues, revenue_per_hl)


def load_model_data(root: str | Path) -> DataBundle:
    return integrate_tables(load_tables(root))


def period_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    week = pd.to_datetime(data["week"])
    masks = {
        "calibration": week.between(CALIBRATION_START, CALIBRATION_END),
        "validation": week.between(VALIDATION_START, VALIDATION_END),
        "development": week.between(CALIBRATION_START, VALIDATION_END),
        "test": week.between(TEST_START, TEST_END),
    }
    if any(masks[a].mul(masks[b]).any() for a, b in [("calibration", "validation"), ("development", "test")]):
        raise AssertionError("Temporal periods overlap")
    return masks
