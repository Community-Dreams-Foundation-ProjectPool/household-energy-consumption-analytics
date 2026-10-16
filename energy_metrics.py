"""Task 9: eight coverage-aware energy metrics from Task 6 aggregate CSVs.

No Streamlit import: calculations are independently testable and reusable.
Only hourly/daily/monthly preprocessed tables are needed, so the Streamlit
app does not repeatedly load the ~2 million minute-level readings.

Energy is OBSERVED energy (sum of available minute readings). Missing
minutes are not filled with zero and no missing-period energy is estimated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = ("hourly", "daily", "monthly")
REQUIRED_COLUMNS = {
    "period_start", "observed_minutes", "expected_minutes",
    "observed_energy_kwh", "mean_active_power_kw", "coverage_pct",
    "is_complete", "sub_metering_1_wh", "sub_metering_2_wh",
    "sub_metering_3_wh",
}
SUBMETER_NAMES = {
    "sub_metering_1_wh": "Kitchen",
    "sub_metering_2_wh": "Laundry",
    "sub_metering_3_wh": "Water heater / AC",
}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}


def validate_aggregates(data: dict[str, pd.DataFrame]) -> None:
    """Reject missing, malformed, or inconsistent analysis CSV data."""
    for level in LEVELS:
        if level not in data:
            raise ValueError(f"Missing {level} dataset")
        df = data[level]
        if df.empty:
            raise ValueError(f"{level} dataset is empty")
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{level} dataset is missing columns: {', '.join(sorted(missing))}")
        if not pd.api.types.is_datetime64_any_dtype(df["period_start"]):
            raise ValueError(f"{level} period_start must contain parsed dates")
        if df["period_start"].isna().any() or df["period_start"].duplicated().any():
            raise ValueError(f"{level} contains invalid or duplicate periods")
        if not df["period_start"].is_monotonic_increasing:
            raise ValueError(f"{level} periods are not sorted")
        numeric_cols = ["observed_minutes", "expected_minutes", "observed_energy_kwh",
                        "mean_active_power_kw", "coverage_pct", *SUBMETER_NAMES]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"{level}.{col} must be numeric")
        for col in ["observed_minutes", "expected_minutes", "observed_energy_kwh", *SUBMETER_NAMES]:
            if df[col].isna().any() or not np.isfinite(df[col]).all() or (df[col] < 0).any():
                raise ValueError(f"{level}.{col} contains missing, negative or non-finite values")
        if (df["observed_minutes"] > df["expected_minutes"]).any():
            raise ValueError(f"{level} has more observed minutes than expected minutes")
        if (df["expected_minutes"] <= 0).any():
            raise ValueError(f"{level} expected minutes must be positive")
        expected_coverage = df["observed_minutes"] / df["expected_minutes"] * 100
        if not np.allclose(df["coverage_pct"], expected_coverage, atol=0.05, rtol=0):
            raise ValueError(f"{level} coverage percentages do not match minute counts")
        if not pd.api.types.is_bool_dtype(df["is_complete"]):
            raise ValueError(f"{level} is_complete must be boolean")
        if not (df["is_complete"] == (df["observed_minutes"] == df["expected_minutes"])).all():
            raise ValueError(f"{level} is_complete disagrees with coverage")
        observed = df["observed_minutes"] > 0
        if df.loc[observed, "mean_active_power_kw"].isna().any():
            raise ValueError(f"{level} has missing mean power for observed periods")
        if not np.isfinite(df.loc[observed, "mean_active_power_kw"]).all():
            raise ValueError(f"{level} has invalid mean power values")
        if (df.loc[observed, "mean_active_power_kw"] < 0).any():
            raise ValueError(f"{level} has negative mean power values")
    # Aggregations should preserve observed kWh across all time granularities.
    totals = [float(data[level]["observed_energy_kwh"].sum()) for level in LEVELS]
    if not np.allclose(totals, totals[0], rtol=1e-7, atol=1e-5):
        raise ValueError("Hourly, daily and monthly observed-energy totals do not match; regenerate CSVs")
    minute_counts = [int(data[level]["observed_minutes"].sum()) for level in LEVELS]
    if len(set(minute_counts)) != 1:
        raise ValueError("Hourly, daily and monthly observed-minute totals do not match")


def compute_hourly_anomalies(hourly: pd.DataFrame) -> pd.DataFrame:
    """Metric 8: hour-of-day IQR outliers, same method as the Task 7 plot.

    Only buckets with at least one observed minute are compared. Partially
    observed hours are included, matching the original Task 7 definition.
    """
    valid = hourly.loc[(hourly["observed_minutes"] > 0) &
                       hourly["mean_active_power_kw"].notna(),
                       ["period_start", "observed_minutes", "coverage_pct", "mean_active_power_kw"]].copy()
    valid["hour_of_day"] = valid["period_start"].dt.hour
    valid["month_start"] = valid["period_start"].dt.to_period("M").dt.to_timestamp()
    if valid.empty:
        valid["anomaly"] = pd.Series(dtype="str")
        return valid
    bounds = valid.groupby("hour_of_day")["mean_active_power_kw"].quantile([0.25, 0.75]).unstack()
    bounds.columns = ["q1", "q3"]
    spread = bounds["q3"] - bounds["q1"]
    bounds["low_bound"] = bounds["q1"] - 1.5 * spread
    bounds["high_bound"] = bounds["q3"] + 1.5 * spread
    valid = valid.join(bounds[["low_bound", "high_bound"]], on="hour_of_day")
    valid["anomaly"] = np.select(
        [valid["mean_active_power_kw"] > valid["high_bound"],
         valid["mean_active_power_kw"] < valid["low_bound"]],
        ["high", "low"], default="normal",
    )
    return valid


def calculate_energy_metrics(data: dict[str, pd.DataFrame]) -> dict:
    """Compute all eight project-defined metrics from preprocessed data.

    Returns small summary values and reusable DataFrames for Streamlit.
    """
    validate_aggregates(data)
    hourly, daily, monthly = (data[level] for level in LEVELS)

    observed_energy = float(monthly["observed_energy_kwh"].sum())
    observed_minutes = int(hourly["observed_minutes"].sum())
    expected_minutes = int(hourly["expected_minutes"].sum())
    mean_kw = observed_energy * 60 / observed_minutes if observed_minutes else float("nan")
    complete_days_pct = float(daily["is_complete"].mean() * 100)

    # Metric 3: compare hours with >=90% data coverage whenever possible.
    candidates = hourly.loc[(hourly["observed_minutes"] > 0) &
                            (hourly["coverage_pct"] >= 90) &
                            hourly["mean_active_power_kw"].notna()]
    used_coverage_fallback = candidates.empty
    if used_coverage_fallback:
        candidates = hourly.loc[(hourly["observed_minutes"] > 0) &
                                hourly["mean_active_power_kw"].notna()]
    peak = candidates.loc[candidates["mean_active_power_kw"].idxmax()] if not candidates.empty else None
    peak_hour = peak["period_start"] if peak is not None else None
    peak_kw = float(peak["mean_active_power_kw"]) if peak is not None else float("nan")
    p95_kw = float(candidates["mean_active_power_kw"].quantile(0.95)) if not candidates.empty else float("nan")

    # Metric 4: weighted mean by observed minutes, not mean of hourly means.
    profile = hourly.assign(hour_of_day=hourly["period_start"].dt.hour).groupby("hour_of_day").agg(
        observed_energy_kwh=("observed_energy_kwh", "sum"),
        observed_minutes=("observed_minutes", "sum"),
    ).reindex(range(24), fill_value=0).reset_index()
    profile["mean_active_power_kw"] = (profile["observed_energy_kwh"] * 60 /
        profile["observed_minutes"].replace(0, np.nan))

    # Metric 5: month/year and meteorological season in Northern Hemisphere.
    month_trend = monthly[["period_start", "observed_energy_kwh", "mean_active_power_kw",
                           "observed_minutes", "coverage_pct"]].copy()
    month_trend["season"] = month_trend["period_start"].dt.month.map(SEASONS)
    seasonal = month_trend.groupby("season").agg(
        observed_energy_kwh=("observed_energy_kwh", "sum"),
        observed_minutes=("observed_minutes", "sum"),
        calendar_months=("period_start", "size"),
    ).reindex(["Winter", "Spring", "Summer", "Fall"]).reset_index()
    seasonal["mean_active_power_kw"] = (seasonal["observed_energy_kwh"] * 60 /
        seasonal["observed_minutes"].replace(0, np.nan))

    # Metric 6: weekday/weekend normalized by observed minutes; raw total is
    # provided separately but should not be used as a like-for-like comparison.
    days = daily.assign(day_of_week=daily["period_start"].dt.dayofweek)
    days = days.assign(day_type=np.where(days["day_of_week"] >= 5, "Weekend", "Weekday"))
    weekday = days.groupby("day_type").agg(
        observed_energy_kwh=("observed_energy_kwh", "sum"),
        observed_minutes=("observed_minutes", "sum"),
        calendar_days=("period_start", "size"),
        complete_days=("is_complete", "sum"),
    ).reindex(["Weekday", "Weekend"]).reset_index()
    weekday["mean_active_power_kw"] = (weekday["observed_energy_kwh"] * 60 /
        weekday["observed_minutes"].replace(0, np.nan))
    weekday["complete_days_pct"] = weekday["complete_days"] / weekday["calendar_days"].replace(0, np.nan) * 100

    # Metric 7: convert kWh to Wh, then calculate three submetering shares and
    # residual 'other' energy. Guard against inconsistent meter sums.
    total_wh = observed_energy * 1000
    category_wh = {name: float(monthly[col].sum()) for col, name in SUBMETER_NAMES.items()}
    residual_wh = total_wh - sum(category_wh.values())
    submeter_inconsistent = residual_wh < -max(1e-6, total_wh * 1e-9)
    category_wh["Other (unmetered)"] = max(residual_wh, 0.0)
    submeter = pd.DataFrame({"category": list(category_wh), "energy_wh": list(category_wh.values())})
    submeter["share_pct"] = submeter["energy_wh"] / total_wh * 100 if total_wh > 0 else np.nan

    # Metric 8: baseline distribution is calculated by hour-of-day (IQR).
    anomalies = compute_hourly_anomalies(hourly)
    anomaly_counts = anomalies["anomaly"].value_counts()
    high_count = int(anomaly_counts.get("high", 0))
    low_count = int(anomaly_counts.get("low", 0))
    anomaly_monthly = (anomalies.loc[anomalies["anomaly"] != "normal"]
        .groupby(["month_start", "anomaly"]).size().unstack(fill_value=0)
        .reindex(columns=["high", "low"], fill_value=0).reset_index())

    return {
        "summary": {
            "observed_energy_kwh": observed_energy,
            "observed_minutes": observed_minutes,
            "expected_minutes": expected_minutes,
            "coverage_pct": observed_minutes / expected_minutes * 100 if expected_minutes else float("nan"),
            "mean_active_power_kw": mean_kw,
            "complete_days_pct": complete_days_pct,
            "incomplete_days": int((~daily["is_complete"]).sum()),
            "peak_hour": peak_hour,
            "peak_hour_mean_kw": peak_kw,
            "peak_p95_kw": p95_kw,
            "peak_fallback": used_coverage_fallback,
            "high_anomaly_hours": high_count,
            "low_anomaly_hours": low_count,
            "evaluated_hours": int(len(anomalies)),
            "submeter_inconsistent": submeter_inconsistent,
        },
        "hourly_profile": profile,
        "monthly_trend": month_trend,
        "seasonal_trend": seasonal,
        "weekday_weekend": weekday,
        "submetering": submeter,
        "hourly_anomalies": anomalies,
        "monthly_anomalies": anomaly_monthly,
    }
