"""Household Energy Analytics Dashboard — Task 10 interactive edition.

   Run `python src/setup_data.py` once to prepare the data, then
   `python -m streamlit run streamlit_app.py`.
Loads preprocessed hourly/daily/monthly CSVs; recomputes all eight metrics for
selected days. The six original plots are retained as unfiltered references.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard_filters import (
    MONTH_NAMES, daily_weekday_profile, filter_aggregates,
    hourly_weekday_heatmap, read_cleaned_minute_preview,
)
from energy_metrics import calculate_energy_metrics, validate_aggregates

BASE = Path(__file__).resolve().parent
PLOTS = BASE / "plots"
DATA_DIR = BASE / "data" / "processed"
AGGREGATE_FILES = {
    "hourly": DATA_DIR / "household_energy_hourly.csv",
    "daily": DATA_DIR / "household_energy_daily.csv",
    "monthly": DATA_DIR / "household_energy_monthly.csv",
}
CLEANED_FILE = DATA_DIR / "household_energy_cleaned.csv"

st.set_page_config(
    page_title="Household Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def read_cached_csv(path_str: str, modified_ns: int, size_bytes: int) -> pd.DataFrame:
    """Cache parsed aggregates, invalidating when file size/mtime changes."""
    df = pd.read_csv(path_str, parse_dates=["period_start"])
    if "is_complete" in df:
        values = df["is_complete"].astype(str).str.strip().str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError(f"Invalid is_complete values in {Path(path_str).name}")
        df["is_complete"] = values.eq("true")
    return df


def load_aggregates() -> tuple[dict[str, pd.DataFrame] | None, str | None]:
    """Load and validate the three full preprocessed datasets."""
    missing = [p.name for p in AGGREGATE_FILES.values() if not p.is_file() or p.stat().st_size == 0]
    if missing:
        return None, "Missing preprocessed CSVs: " + ", ".join(missing)
    try:
        data = {
            name: read_cached_csv(str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for name, path in AGGREGATE_FILES.items()
        }
        validate_aggregates(data)
        return data, None
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        return None, f"Preprocessed data could not be loaded: {exc}"


@st.cache_data(show_spinner="Calculating selected energy metrics...")
def cached_energy_metrics(hourly: pd.DataFrame, daily: pd.DataFrame,
                          monthly: pd.DataFrame) -> dict:
    return calculate_energy_metrics({"hourly": hourly, "daily": daily, "monthly": monthly})


@st.cache_data(show_spinner="Reading a limited preview of cleaned minute-level data...")
def cached_minute_preview(path: str, modified_ns: int, size_bytes: int,
                          selected_days: tuple[date, ...], limit: int) -> pd.DataFrame:
    return read_cleaned_minute_preview(path, selected_days, limit=limit)


def clear_filters() -> None:
    for key in ("energy_date_range", "energy_year", "energy_month", "energy_season",
                "energy_day_type"):
        st.session_state.pop(key, None)


def build_sidebar(data: dict[str, pd.DataFrame] | None) -> tuple[bool, dict[str, pd.DataFrame] | None]:
    """Keep the user-requested Energy Explorer sidebar, adding live filters."""
    with st.sidebar:
        st.markdown("## ⚡ Energy Explorer")
        st.caption("Explore household electricity consumption by date and season.")
        st.divider()
        show_explanations = st.toggle(
            "Show chart explanations", value=True,
            help="Display brief interpretations and calculation notes under charts.",
        )
        if data is None:
            st.info("Preprocessed data is not ready. Bundled historical charts are available.")
            st.caption("Run python src/setup_data.py to download and prepare the full UCI dataset.")
            st.divider()
            st.caption("Source: UCI Machine Learning Repository · CC BY 4.0")
            return show_explanations, None

        daily = data["daily"]
        min_day = daily["period_start"].min().date()
        max_day = daily["period_start"].max().date()
        st.markdown("### Filter consumption")
        st.button("↺ Reset filters", on_click=clear_filters, use_container_width=True)
        dates = st.date_input(
            "Date range (inclusive)", value=(min_day, max_day),
            min_value=min_day, max_value=max_day, key="energy_date_range",
        )
        years = sorted(daily["period_start"].dt.year.unique().tolist())
        year_label = st.selectbox("Year", ["All years", *years], key="energy_year")
        month_label = st.selectbox("Month", ["All months", *MONTH_NAMES], key="energy_month")
        season_label = st.selectbox(
            "Season", ["All seasons", "Winter", "Spring", "Summer", "Fall"],
            key="energy_season",
        )
        day_type = st.selectbox(
            "Day type", ["All days", "Weekdays", "Weekends"], key="energy_day_type",
        )
        if not isinstance(dates, (tuple, list)) or len(dates) != 2:
            st.warning("Select both a start and an end date to apply filters.")
            return show_explanations, None
        start, end = dates
        if start > end:
            st.warning("The start date must not be later than the end date.")
            return show_explanations, None
        try:
            filtered = filter_aggregates(
                data, start, end,
                year=None if year_label == "All years" else int(year_label),
                month=None if month_label == "All months" else MONTH_NAMES.index(month_label) + 1,
                season=None if season_label == "All seasons" else season_label,
                day_type=day_type,
            )
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Could not apply filters: {exc}")
            filtered = None
        if filtered is None:
            st.warning("No calendar days match this filter combination.")
        else:
            observed = int(filtered["daily"]["observed_minutes"].sum())
            st.caption(f"{len(filtered['daily']):,} selected days · {observed:,} observed minute readings")
            if observed == 0:
                st.warning("Selected days contain no valid minute readings.")
        st.divider()
        st.markdown("**Dataset**")
        st.caption("Sceaux, France · Dec 2006–Nov 2010")
        st.caption("One-minute readings · UCI Machine Learning Repository · CC BY 4.0")
    return show_explanations, filtered


def explain(message: str, enabled: bool) -> None:
    if enabled:
        st.caption(message)


def show_overview(metrics: dict | None, selected: dict[str, pd.DataFrame] | None) -> None:
    st.subheader("Energy consumption overview")
    c1, c2, c3, c4 = st.columns(4)
    if metrics is None:
        c1.metric("Original records (documented)", "2,075,259")
        c2.metric("Valid records (documented)", "2,049,280")
        c3.metric("Missing records removed", "25,979")
        c4.metric("Bundled visualizations", "6")
        st.info("Select a nonempty period with observed readings to view live metrics. "
                "The original plots remain in **Original plots**.")
        return
    summary = metrics["summary"]
    c1.metric("Observed energy", f"{summary['observed_energy_kwh']:,.0f} kWh")
    c2.metric("Average active power", f"{summary['mean_active_power_kw']:.2f} kW")
    peak = summary["peak_hour_mean_kw"]
    c3.metric("Peak hourly mean", f"{peak:.2f} kW" if pd.notna(peak) else "N/A")
    c4.metric("Complete selected days", f"{summary['complete_days_pct']:.1f}%")
    if selected is not None:
        days = selected["daily"]["period_start"]
        st.caption(f"Showing {len(days):,} selected calendar days · "
                   f"{days.min():%d %b %Y} to {days.max():%d %b %Y} · "
                   f"{summary['coverage_pct']:.2f}% minute coverage")
    st.caption("Energy is the sum of observed minute readings only. Missing minutes are "
               "not zero-filled or estimated. Partial days/months are not full-period totals.")


def show_trends(data: dict[str, pd.DataFrame] | None, enabled: bool) -> None:
    st.subheader("Energy consumption trends")
    if data is None or int(data["daily"]["observed_minutes"].sum()) == 0:
        st.info("No observed energy in this selection. Adjust the filters or view Original plots.")
        return
    daily, monthly = data["daily"], data["monthly"]
    st.markdown("#### Daily observed energy")
    st.line_chart(daily, x="period_start", y="observed_energy_kwh", use_container_width=True)
    n_incomplete = int((~daily["is_complete"]).sum())
    explain(f"Daily energy (kWh) from available readings. {n_incomplete:,} of "
            f"{len(daily):,} selected days are incomplete; avoid interpreting "
            "their observed totals as full-day consumption. · Metric 1", enabled)
    st.markdown("#### Monthly average active power")
    st.line_chart(monthly, x="period_start", y="mean_active_power_kw", use_container_width=True)
    explain("Monthly average power (kW), weighted by observed minutes. Each point "
            "uses only days selected by the filters, even for partial months. · Metrics 2 & 5", enabled)
    with st.expander("Monthly observed energy (kWh)", expanded=False):
        st.bar_chart(monthly, x="period_start", y="observed_energy_kwh", use_container_width=True)
        st.caption("Month buckets can contain only part of a calendar month when filters are active.")


def show_patterns(data: dict[str, pd.DataFrame] | None, metrics: dict | None,
                  enabled: bool) -> None:
    st.subheader("Consumption patterns")
    if data is None or metrics is None:
        st.info("Choose filters that include observed energy readings.")
        return
    st.markdown("#### Typical active power by day of week")
    profile = daily_weekday_profile(data["daily"])
    st.bar_chart(profile, x="weekday", y="mean_active_power_kw", use_container_width=True)
    explain("Average power by weekday is weighted by available minute readings. "
            "This supports fair comparisons despite unequal day counts. · Metric 6", enabled)
    st.markdown("#### Peak-hour heatmap")
    matrix = hourly_weekday_heatmap(data["hourly"])
    if matrix.notna().any().any():
        fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
        heat = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
                         aspect="auto", interpolation="nearest", cmap="YlOrRd")
        ax.set_xticks(range(24))
        ax.set_xticklabels(range(24), fontsize=8)
        ax.set_yticks(range(7))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Day of week")
        ax.set_title("Average active power by weekday and hour")
        fig.colorbar(heat, ax=ax, label="Mean active power (kW)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Not enough observed readings to display a heatmap.")
    explain("Cells show mean active power weighted by available minutes. "
            "Empty weekday/hour combinations remain blank. · Metrics 3 & 4", enabled)
    st.markdown("#### Weekday versus weekend")
    st.bar_chart(metrics["weekday_weekend"], x="day_type", y="mean_active_power_kw",
                 use_container_width=True)
    explain("Mean kW is comparable across groups with different numbers of days. · Metric 6", enabled)


def show_advanced(metrics: dict | None, enabled: bool) -> None:
    st.subheader("Advanced energy metrics")
    if metrics is None:
        st.info("No metrics are available for this selection.")
        return
    st.markdown("#### Sub-metering category shares")
    st.bar_chart(metrics["submetering"], x="category", y="share_pct", use_container_width=True)
    if metrics["summary"]["submeter_inconsistent"]:
        st.warning("Sub-meter readings exceed total measured energy; residual 'Other' is clipped to zero.")
    explain("Shares of observed energy assigned to kitchen, laundry, water heater/AC "
            "and unmetered uses. · Metric 7", enabled)
    st.markdown("#### Unusual consumption hours by month")
    anomalies = metrics["monthly_anomalies"]
    if anomalies.empty:
        st.info("No high or low anomaly hours were flagged for the selected data.")
    else:
        st.bar_chart(anomalies, x="month_start", y=["high", "low"], use_container_width=True)
    explain("Anomaly thresholds are recomputed from the selected hours using "
            "hour-of-day IQR bounds. Short selections can give unstable baselines; "
            "flags do not prove a cause. · Metric 8", enabled)


def show_calculated_metrics(metrics: dict | None) -> None:
    st.subheader("Calculated energy metrics")
    if metrics is None:
        st.warning("Select a period containing observed readings to calculate metrics.")
        return
    summary = metrics["summary"]
    st.success("All eight metrics are recalculated for your selection and cached for fast reruns.")
    st.markdown("#### 1. Total observed energy · 2. Mean active power")
    a, b, c = st.columns(3)
    a.metric("Observed energy", f"{summary['observed_energy_kwh']:,.2f} kWh")
    b.metric("Mean active power", f"{summary['mean_active_power_kw']:.3f} kW")
    c.metric("Observed-minute coverage", f"{summary['coverage_pct']:.2f}%")
    st.caption("Energy = Σ (minute-level kW / 60). Mean power is weighted by observed minutes.")
    st.markdown("#### 3. Peak-demand period")
    a, b, c = st.columns(3)
    peak = summary["peak_hour_mean_kw"]
    p95 = summary["peak_p95_kw"]
    a.metric("Peak hourly mean", f"{peak:.2f} kW" if pd.notna(peak) else "N/A")
    b.metric("95th-percentile hourly mean", f"{p95:.2f} kW" if pd.notna(p95) else "N/A")
    c.metric("Peak hour", summary["peak_hour"].strftime("%d %b %Y %H:%M")
             if summary["peak_hour"] is not None else "N/A")
    st.caption("Prefers hours with at least 90% observed-minute coverage when available.")
    st.markdown("#### 4. Time-of-day load profile")
    st.line_chart(metrics["hourly_profile"], x="hour_of_day", y="mean_active_power_kw")
    st.markdown("#### 5. Seasonal and monthly trends")
    st.line_chart(metrics["monthly_trend"], x="period_start", y="mean_active_power_kw")
    st.dataframe(metrics["seasonal_trend"].round(3), hide_index=True, use_container_width=True)
    st.markdown("#### 6. Weekday versus weekend")
    st.dataframe(metrics["weekday_weekend"].round(3), hide_index=True, use_container_width=True)
    st.markdown("#### 7. Sub-metering shares")
    st.dataframe(metrics["submetering"].round(3), hide_index=True, use_container_width=True)
    st.markdown("#### 8. Unusual consumption flags")
    a, b, c = st.columns(3)
    a.metric("High anomaly hours", f"{summary['high_anomaly_hours']:,}")
    b.metric("Low anomaly hours", f"{summary['low_anomaly_hours']:,}")
    c.metric("Evaluated hours", f"{summary['evaluated_hours']:,}")
    st.caption("Anomalies use selected hourly data and hour-of-day IQR bounds; "
               "a flag does not establish a cause.")


def show_data_explorer(data: dict[str, pd.DataFrame] | None) -> None:
    st.subheader("Explore and export processed data")
    if data is None:
        st.warning("No selected data to display. Adjust the sidebar filters.")
        return
    granularity = st.selectbox("Data granularity", ["Daily", "Hourly", "Monthly"])
    df = data[granularity.lower()]
    st.caption(f"{len(df):,} {granularity.lower()} rows match your filters. "
               "The table displays up to the first 2,000 rows; the CSV download includes all selected rows.")
    st.dataframe(df.head(2000), use_container_width=True, hide_index=True, height=380)
    st.download_button(
        f"Download selected {granularity.lower()} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"household_energy_selected_{granularity.lower()}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.divider()
    st.markdown("#### Preview cleaned minute-level readings")
    st.caption("The full cleaned minute-level file is large. This optional preview "
               "reads at most 2,000 matching rows without loading the entire dataset into memory.")
    if not CLEANED_FILE.is_file():
        st.info("The cleaned minute-level CSV is not present. Run `python src/setup_data.py` "
                "to prepare it. Aggregated tables remain available.")
        return
    if st.checkbox("Load up to 2,000 matching minute readings", value=False,
                   key="energy_show_minute_preview"):
        selected_days = tuple(data["daily"]["period_start"].dt.date)
        try:
            raw = cached_minute_preview(
                str(CLEANED_FILE), CLEANED_FILE.stat().st_mtime_ns,
                CLEANED_FILE.stat().st_size, selected_days, 2000,
            )
            if raw.empty:
                st.info("No cleaned minute rows match these filters.")
            else:
                st.dataframe(raw, use_container_width=True, hide_index=True, height=360)
                st.download_button(
                    "Download minute-level preview CSV", raw.to_csv(index=False).encode("utf-8"),
                    file_name="household_energy_selected_minute_preview.csv", mime="text/csv",
                )
                st.caption("These are the earliest matching rows, not a random sample "
                           "and not the complete selected minute-level dataset.")
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            st.error(f"Could not load cleaned minute-level preview: {exc}")


def show_quality(data: dict[str, pd.DataFrame] | None, metrics: dict | None) -> None:
    st.subheader("Data quality and methods")
    st.markdown("Source: UCI Individual Household Electric Power Consumption "
                "(one household in Sceaux, France, 2006–2010). The cleaning stage "
                "removed **25,979** missing-measurement rows rather than imputing them.")
    if data is not None and metrics is not None:
        s = metrics["summary"]
        a, b, c = st.columns(3)
        a.metric("Selected observed minutes", f"{s['observed_minutes']:,}")
        b.metric("Selected incomplete days", f"{s['incomplete_days']:,}")
        c.metric("Selected minute coverage", f"{s['coverage_pct']:.2f}%")
        st.caption("Coverage denominators reflect selected calendar days/hours. "
                   "For filtered months, totals and coverage use selected days only, "
                   "not the whole calendar month.")
    else:
        st.info("Selected-period data quality metrics are unavailable.")
    st.markdown("**Calculation methods**")
    st.markdown(
        "- **Observed energy (kWh):** sum of available one-minute kW readings divided by 60.\n"
        "- **Mean active power (kW):** observed energy × 60 / observed minutes.\n"
        "- **Coverage (%):** observed minutes / expected minutes × 100.\n"
        "- **Sub-metering shares:** each category's observed Wh divided by total observed Wh.\n"
        "- **Anomalies:** hourly mean outside hour-of-day 1.5×IQR bounds."
    )
    st.info("A partial period is not a full-period energy estimate. Anomaly flags "
            "are statistical, not proof of an appliance fault.")
    findings = BASE / "docs" / "Household_Energy_Visualization_Findings.docx"
    if findings.is_file():
        st.download_button(
            "Download visualization findings (DOCX)", data=findings.read_bytes(),
            file_name=findings.name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


def show_reference_plot(filename: str, heading: str, description: str,
                        metric: str, enabled: bool) -> None:
    st.markdown(f"#### {heading}")
    path = PLOTS / filename
    if path.is_file():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Bundled plot missing: {filename}")
    explain(f"{description} · {metric}", enabled)


def show_original_plots(enabled: bool) -> None:
    st.subheader("Original project visualizations")
    st.info("These six Task 7 plots were generated for the **full original dataset**. "
            "They do **not** respond to the sidebar filters. Use the other tabs "
            "for live filtered charts.")
    items = [
        ("01_daily_total_energy_trend.png", "Daily observed energy",
         "Historical daily observed kWh; incomplete days are marked.", "Metric 1"),
        ("02_monthly_avg_power_trend.png", "Monthly average active power",
         "Historical monthly demand and seasonal variation.", "Metrics 2 & 5"),
        ("03_avg_consumption_by_day_of_week.png", "Average by day of week",
         "Historical average weekday/weekend demand.", "Metric 6"),
        ("04_hour_by_weekday_heatmap.png", "Peak-hour heatmap",
         "Historical weekday/hour consumption profile.", "Metrics 3 & 4"),
        ("05_submetering_category_share.png", "Sub-metering shares",
         "Historical allocation of observed energy.", "Metric 7"),
        ("06_anomalous_hours_by_month.png", "Unusual hours",
         "Historical counts of flagged unusual consumption hours.", "Metric 8"),
    ]
    for filename, heading, description, metric in items:
        show_reference_plot(filename, heading, description, metric, enabled)


def main() -> None:
    st.title("⚡ Household Energy Consumption Analytics Dashboard")
    st.caption("Interactive household electricity insights · France, 2006–2010")
    source, error = load_aggregates()
    show_explanations, filtered = build_sidebar(source)
    metrics = None
    if filtered is not None and int(filtered["hourly"]["observed_minutes"].sum()) > 0:
        try:
            metrics = cached_energy_metrics(
                filtered["hourly"], filtered["daily"], filtered["monthly"]
            )
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            error = f"Energy metric calculation failed: {exc}"
    if error:
        st.warning(error)
    show_overview(metrics, filtered)
    st.divider()
    trend_tab, pattern_tab, advanced_tab, metric_tab, explorer_tab, quality_tab, originals_tab = st.tabs(
        ["Consumption trends", "Usage patterns", "Advanced metrics", "Calculated metrics",
         "Data explorer", "Data quality", "Original plots"]
    )
    with trend_tab:
        show_trends(filtered, show_explanations)
    with pattern_tab:
        show_patterns(filtered, metrics, show_explanations)
    with advanced_tab:
        show_advanced(metrics, show_explanations)
    with metric_tab:
        show_calculated_metrics(metrics)
    with explorer_tab:
        show_data_explorer(filtered)
    with quality_tab:
        show_quality(filtered, metrics)
    with originals_tab:
        show_original_plots(show_explanations)


if __name__ == "__main__":
    main()
