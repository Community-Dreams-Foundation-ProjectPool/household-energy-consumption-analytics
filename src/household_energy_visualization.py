from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "plots"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

sns.set_theme(style="whitegrid")



# Data loading
def load_inputs(base: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the four CSVs produced by the Task 6 aggregation script."""
    paths = {
        "featured": base / "household_energy_featured.csv",
        "hourly": base / "household_energy_hourly.csv",
        "daily": base / "household_energy_daily.csv",
        "monthly": base / "household_energy_monthly.csv",
    }
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing input file(s) for: {missing}. Run the Task 6 aggregation "
            "script first to produce them."
        )
    data = {name: pd.read_csv(p) for name, p in paths.items()}
    for key in ("hourly", "daily", "monthly"):
        data[key]["period_start"] = pd.to_datetime(data[key]["period_start"])
    return data

# Plot 1: Line plot - daily total energy trend (Metric 1)
def plot_daily_energy_trend(daily: pd.DataFrame) -> plt.Figure:
    """Line plot of observed daily energy (kWh), flagging incomplete days.

    Metric 1: Total Energy Consumption, at daily granularity.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        daily["period_start"], daily["observed_energy_kwh"],
        color="#2b6cb0", linewidth=1, label="Observed daily energy (kWh)",
    )

    incomplete = daily[~daily["is_complete"]]
    if not incomplete.empty:
        ax.scatter(
            incomplete["period_start"], incomplete["observed_energy_kwh"],
            color="#e53e3e", s=18, zorder=3,
            label="Incomplete day (coverage < 100%)",
        )

    ax.set_title("Daily Total Energy Consumption (Observed)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (kWh)")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

# Plot 2: Line plot - monthly average active power trend (Metric 2)
def plot_monthly_avg_power_trend(monthly: pd.DataFrame) -> plt.Figure:
    """Line plot of mean active power (kW) by month.

    Metric 2: Average Consumption (Mean Active Power) - a rate, distinct
    from Metric 1's cumulative kWh total, tracked at monthly granularity.
    """
    df = monthly.sort_values("period_start")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df["period_start"], df["mean_active_power_kw"],
        color="#38a169", marker="o", markersize=4, linewidth=1.5,
    )
    ax.set_title("Average Active Power by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Active Power (kW)")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# Plot 3: Bar chart - average consumption by day of week (Metric 6)

def plot_avg_by_day_of_week(featured: pd.DataFrame) -> plt.Figure:
    """Bar chart of mean active power by day of week, weekend bars highlighted.

    Metric 6: Weekday vs. Weekend Consumption Comparison.
    """
    by_day = (
        featured.groupby("day_name")["global_active_power_kw"]
        .mean()
        .reindex(DAY_ORDER)
    )
    colors = ["#2b6cb0" if d not in ("Saturday", "Sunday") else "#dd6b20" for d in DAY_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(by_day.index, by_day.values, color=colors)
    ax.set_title("Average Active Power by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean Active Power (kW)")
    ax.tick_params(axis="x", rotation=30)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2b6cb0", label="Weekday"),
        plt.Rectangle((0, 0), 1, 1, color="#dd6b20", label="Weekend"),
    ]
    ax.legend(handles=handles)
    fig.tight_layout()
    return fig


# Plot 4: Heatmap - hour of day x day of week (Metrics 3 & 4)

def plot_hour_by_weekday_heatmap(featured: pd.DataFrame) -> plt.Figure:
    """Heatmap of mean active power by hour-of-day and day-of-week.

    Metric 4: Time-of-Day Load Profile directly; also supports Metric 3
    (Peak-Demand Period) by making the peak hour/day cell visible at a glance.
    """
    pivot = (
        featured.pivot_table(
            index="hour", columns="day_name",
            values="global_active_power_kw", aggfunc="mean",
        )
        .reindex(columns=DAY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(
        pivot, cmap="YlOrRd", ax=ax,
        cbar_kws={"label": "Mean Active Power (kW)"},
    )
    ax.set_title("Mean Active Power by Hour of Day and Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    ax.set_yticks(np.arange(0, 24, 2) + 0.5)
    ax.set_yticklabels(range(0, 24, 2), rotation=0)
    fig.tight_layout()
    return fig

# Plot 5: Stacked bar - sub-metering category share by month (Metric 7)

def compute_submetering_shares(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return each sub-metering category's % share of total energy, by month.

    Shared calculation used by both the plot below and, later, by the
    Streamlit app's metric table.
    """
    df = monthly.copy()
    total_wh = df["observed_energy_kwh"] * 1000.0
    other_wh = (
        total_wh
        - df["sub_metering_1_wh"] - df["sub_metering_2_wh"] - df["sub_metering_3_wh"]
    ).clip(lower=0)

    shares = pd.DataFrame({
        "period_start": df["period_start"],
        "Kitchen": df["sub_metering_1_wh"],
        "Laundry": df["sub_metering_2_wh"],
        "Water Heater / AC": df["sub_metering_3_wh"],
        "Other (unmetered)": other_wh,
    })
    category_cols = ["Kitchen", "Laundry", "Water Heater / AC", "Other (unmetered)"]
    shares[category_cols] = shares[category_cols].div(total_wh.replace(0, np.nan), axis=0) * 100
    return shares


def plot_submetering_share(monthly: pd.DataFrame) -> plt.Figure:
    """Stacked bar chart of sub-metering category share of total energy, by month.

    Metric 7: Sub-Metering Category Share (kitchen, laundry, water heater/AC,
    and unmetered "other" as a percentage of total energy).
    """
    shares = compute_submetering_shares(monthly)
    plot_df = shares.set_index(shares["period_start"].dt.strftime("%Y-%m")).drop(columns="period_start")

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df.plot(kind="bar", stacked=True, ax=ax,
                 color=["#2b6cb0", "#38a169", "#dd6b20", "#a0aec0"])
    ax.set_title("Sub-Metering Category Share of Total Energy, by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of Total Energy (%)")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticks(ax.get_xticks()[::2])  # thin out labels; 48 months is too dense otherwise
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# Plot 6: Stacked bar - anomalous hours per month (Metric 8)

def compute_hourly_anomalies(hourly: pd.DataFrame) -> pd.DataFrame:
    """Flag each hourly reading as a high/low anomaly using an IQR bound
    computed per hour-of-day (so a normal midday load isn't compared against
    a normal midnight load). Mirrors Metric 8's calculation method.

    Hours with zero observed minutes (fully missing) are excluded from both
    the baseline calculation and the flagging, since there is no reading to
    flag.
    """
    df = hourly.copy()
    df["hour_of_day"] = df["period_start"].dt.hour
    df["month_start"] = df["period_start"].dt.to_period("M").dt.to_timestamp()

    valid = df[df["observed_minutes"] > 0].copy()
    bounds = (
        valid.groupby("hour_of_day")["mean_active_power_kw"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.75: "q3"})
    )
    bounds["iqr"] = bounds["q3"] - bounds["q1"]
    bounds["low_bound"] = bounds["q1"] - 1.5 * bounds["iqr"]
    bounds["high_bound"] = bounds["q3"] + 1.5 * bounds["iqr"]

    valid = valid.merge(bounds[["low_bound", "high_bound"]], on="hour_of_day", how="left")
    valid["anomaly"] = np.select(
        [valid["mean_active_power_kw"] > valid["high_bound"],
         valid["mean_active_power_kw"] < valid["low_bound"]],
        ["high", "low"],
        default="normal",
    )
    return valid


def plot_anomalous_hours_by_month(hourly: pd.DataFrame) -> plt.Figure:
    """Stacked bar chart of the count of high/low anomalous hours per month.

    Metric 8: Unusual Consumption / Anomaly Flag.
    """
    flagged = compute_hourly_anomalies(hourly)
    counts = (
        flagged[flagged["anomaly"] != "normal"]
        .groupby(["month_start", "anomaly"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["high", "low"], fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#e53e3e", "#3182ce"])
    ax.set_title("Anomalous Hours per Month (IQR by Hour-of-Day Baseline)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Anomalous Hours")
    ax.set_xticklabels([d.strftime("%Y-%m") for d in counts.index], rotation=45)
    ax.set_xticks(ax.get_xticks()[::2])
    ax.legend(title="Anomaly Type", labels=["High spike", "Low drop"])
    fig.tight_layout()
    return fig


# Registry + orchestration
def generate_all_figures(data: dict[str, pd.DataFrame]) -> dict[str, plt.Figure]:
    """Build every figure and return them keyed by output filename stem.

    This is the function the Streamlit app should import and call directly
    (selecting whichever figures it needs) rather than re-implementing any
    plotting logic.
    """
    return {
        "01_daily_total_energy_trend": plot_daily_energy_trend(data["daily"]),
        "02_monthly_avg_power_trend": plot_monthly_avg_power_trend(data["monthly"]),
        "03_avg_consumption_by_day_of_week": plot_avg_by_day_of_week(data["featured"]),
        "04_hour_by_weekday_heatmap": plot_hour_by_weekday_heatmap(data["featured"]),
        "05_submetering_category_share": plot_submetering_share(data["monthly"]),
        "06_anomalous_hours_by_month": plot_anomalous_hours_by_month(data["hourly"]),
    }


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    data = load_inputs()
    figures = generate_all_figures(data)

    print("Plots written:")
    for name, fig in figures.items():
        path = save_figure(fig, PLOTS_DIR / f"{name}.png")
        print(f"  - {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
