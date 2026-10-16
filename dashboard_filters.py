"""Task 10: fast, testable filters and consistent reaggregation.

Filters select complete calendar DAYS from preprocessed data. Monthly totals
are rebuilt from those selected days, so a partial-month or weekday-only
selection never accidentally includes excluded days. Missing minutes are
never filled, estimated, or treated as zero consumption.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from energy_metrics import SEASONS, validate_aggregates


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def filter_aggregates(
    data: dict[str, pd.DataFrame],
    start_date: date,
    end_date: date,
    year: int | None = None,
    month: int | None = None,
    season: str | None = None,
    day_type: str = "All days",
) -> dict[str, pd.DataFrame] | None:
    """Filter calendar days, retain corresponding hours, rebuild monthly totals.

    Returns None for an empty selection, not malformed empty metric tables.
    Assumes the original source aggregates have already been validated.
    """
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date")
    if season is not None and season not in {"Winter", "Spring", "Summer", "Fall"}:
        raise ValueError(f"Unknown season: {season}")
    if day_type not in {"All days", "Weekdays", "Weekends"}:
        raise ValueError(f"Unknown day type: {day_type}")
    if month is not None and month not in range(1, 13):
        raise ValueError("Month must be between 1 and 12")

    daily = data["daily"]
    days = daily["period_start"]
    mask = (days.dt.date >= start_date) & (days.dt.date <= end_date)
    if year is not None:
        mask &= days.dt.year.eq(year)
    if month is not None:
        mask &= days.dt.month.eq(month)
    if season is not None:
        mask &= days.dt.month.map(SEASONS).eq(season)
    if day_type == "Weekdays":
        mask &= days.dt.dayofweek.lt(5)
    elif day_type == "Weekends":
        mask &= days.dt.dayofweek.ge(5)
    selected_daily = daily.loc[mask].copy().reset_index(drop=True)
    if selected_daily.empty:
        return None

    # Keep hourly buckets for exactly the selected calendar days.
    selected_dates = pd.DatetimeIndex(selected_daily["period_start"])
    hourly = data["hourly"]
    selected_hourly = hourly.loc[
        hourly["period_start"].dt.normalize().isin(selected_dates)
    ].copy().reset_index(drop=True)

    # Monthly results are reaggregated from SELECTED days, not sliced from
    # full-calendar-month summaries. This is essential for weekday/season/date
    # filters that select only part of a month.
    selected_daily["month_start"] = selected_daily["period_start"].dt.to_period("M").dt.to_timestamp()
    totals = ["observed_minutes", "expected_minutes", "observed_energy_kwh",
              "sub_metering_1_wh", "sub_metering_2_wh", "sub_metering_3_wh"]
    grouped = selected_daily.groupby("month_start", sort=True)
    monthly = grouped[totals].sum().reset_index().rename(columns={"month_start": "period_start"})
    monthly["mean_active_power_kw"] = (
        monthly["observed_energy_kwh"] * 60 /
        monthly["observed_minutes"].replace(0, np.nan)
    )
    if "max_minute_active_power_kw" in selected_daily:
        max_power = grouped["max_minute_active_power_kw"].max().to_numpy()
        monthly["max_minute_active_power_kw"] = max_power
    monthly["coverage_pct"] = monthly["observed_minutes"] / monthly["expected_minutes"] * 100
    monthly["is_complete"] = monthly["observed_minutes"].eq(monthly["expected_minutes"])
    selected_daily = selected_daily.drop(columns=["month_start"])

    result = {"hourly": selected_hourly, "daily": selected_daily, "monthly": monthly}
    validate_aggregates(result)
    return result


def hourly_weekday_heatmap(hourly: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean kW for weekday/hour cells, no zero-fill of empty cells."""
    selected = hourly.copy()
    selected["day_of_week"] = selected["period_start"].dt.dayofweek
    selected["hour"] = selected["period_start"].dt.hour
    grouped = selected.groupby(["day_of_week", "hour"], sort=True)[
        ["observed_energy_kwh", "observed_minutes"]
    ].sum()
    power = grouped["observed_energy_kwh"] * 60 / grouped["observed_minutes"].replace(0, np.nan)
    return power.unstack("hour").reindex(index=range(7), columns=range(24))


def daily_weekday_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean kW by day of week, not biased by missing minutes."""
    selected = daily.copy()
    selected["day_of_week"] = selected["period_start"].dt.dayofweek
    grouped = selected.groupby("day_of_week")[["observed_energy_kwh", "observed_minutes"]].sum()
    grouped = grouped.reindex(range(7))
    grouped["mean_active_power_kw"] = grouped["observed_energy_kwh"] * 60 / grouped["observed_minutes"].replace(0, np.nan)
    grouped["weekday"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return grouped.reset_index()[["weekday", "mean_active_power_kw", "observed_minutes"]]


def read_cleaned_minute_preview(path: str, selected_days: tuple[date, ...],
                                limit: int = 2000) -> pd.DataFrame:
    """Read only the first matching cleaned minute rows, in bounded chunks.

    This deliberately avoids loading the full ~2-million-row CSV into the
    Streamlit process. Results are a PREVIEW, not a representative sample.
    """
    if not selected_days or limit <= 0:
        return pd.DataFrame()
    allowed = set(selected_days)
    output = []
    count = 0
    reader = pd.read_csv(path, chunksize=100_000, parse_dates=["timestamp"])
    try:
        for chunk in reader:
            subset = chunk.loc[chunk["timestamp"].dt.date.isin(allowed)]
            if subset.empty:
                continue
            portion = subset.head(limit - count)
            output.append(portion)
            count += len(portion)
            if count >= limit:
                break
    finally:
        reader.close()
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()
