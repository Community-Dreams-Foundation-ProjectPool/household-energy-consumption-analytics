"""Task 6: feature engineering and coverage-aware energy aggregation.

Run: python household_energy_aggregation.py
Input: data/processed/household_energy_cleaned.csv (output of the cleaning script)
Outputs: data/processed/household_energy_featured.csv, data/processed/household_energy_hourly.csv,
         data/processed/household_energy_daily.csv, data/processed/household_energy_monthly.csv
Power (kW) is minute-averaged. One observed minute contributes kW / 60 kWh.
Missing minutes are NOT imputed or treated as zero. Summed energy is therefore
observed energy, not an estimate of full-period consumption. Coverage is
reported to identify incomplete periods. Hourly/daily/monthly averages use
observed samples only. Monthly aggregation sums daily energy and counts.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT = BASE / 'household_energy_cleaned.csv'
MEASUREMENTS = [
    'global_active_power_kw', 'global_reactive_power_kw', 'voltage_v',
    'global_intensity_a', 'sub_metering_1_wh', 'sub_metering_2_wh',
    'sub_metering_3_wh',
]


def season_from_month(month: pd.Series) -> pd.Series:
    """Northern-hemisphere meteorological seasons (dataset is in France)."""
    mapping = {12: 'Winter', 1: 'Winter', 2: 'Winter',
               3: 'Spring', 4: 'Spring', 5: 'Spring',
               6: 'Summer', 7: 'Summer', 8: 'Summer',
               9: 'Fall', 10: 'Fall', 11: 'Fall'}
    return month.map(mapping)


def load_featured(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run Task 4 cleaning script first.')
    df = pd.read_csv(path, parse_dates=['timestamp'])
    missing = {'timestamp', *MEASUREMENTS} - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df['timestamp'].isna().any():
        raise ValueError('Invalid timestamps found in cleaned data.')
    for column in MEASUREMENTS:
        df[column] = pd.to_numeric(df[column], errors='raise')
    if df[MEASUREMENTS].isna().any().any():
        raise ValueError('Cleaned input still contains missing measurements.')
    if df['timestamp'].duplicated().any():
        raise ValueError('Duplicate timestamps: resolve in Task 4 first.')
    if not df['timestamp'].dt.second.eq(0).all() or not df['timestamp'].dt.microsecond.eq(0).all():
        raise ValueError('Expected timestamps aligned to exact minutes.')
    df = df.sort_values('timestamp').reset_index(drop=True)
    t = df['timestamp'].dt
    df['year'] = t.year
    df['month'] = t.month
    df['day'] = t.day
    df['hour'] = t.hour
    df['day_of_week'] = t.dayofweek  # Monday=0, Sunday=6
    df['day_name'] = t.day_name()
    df['is_weekend'] = df['day_of_week'].ge(5)
    df['season'] = season_from_month(df['month'])
    df['date'] = t.normalize()
    df['energy_kwh'] = df['global_active_power_kw'] / 60.0
    df['total_active_energy_wh'] = df['energy_kwh'] * 1000.0
    return df


def aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Observed energy and coverage for full calendar buckets, including empty buckets."""
    indexed = df.set_index('timestamp')
    grouped = indexed.resample(freq)
    result = grouped.agg(
        observed_minutes=('energy_kwh', 'count'),
        observed_energy_kwh=('energy_kwh', 'sum'),
        mean_active_power_kw=('global_active_power_kw', 'mean'),
        max_minute_active_power_kw=('global_active_power_kw', 'max'),
        sub_metering_1_wh=('sub_metering_1_wh', 'sum'),
        sub_metering_2_wh=('sub_metering_2_wh', 'sum'),
        sub_metering_3_wh=('sub_metering_3_wh', 'sum'),
    )
    result.index.name = 'period_start'
    # A calendar period can contain 23/25 hours around DST, so use the next
    # actual period boundary rather than assuming 24 hours for every day.
    if freq == 'h':
        next_starts = result.index + pd.DateOffset(hours=1)
    elif freq == 'D':
        next_starts = result.index + pd.DateOffset(days=1)
    elif freq == 'MS':
        next_starts = result.index + pd.DateOffset(months=1)
    else:
        raise ValueError('Supported frequencies: h, D, MS')
    # Timestamps in this dataset are naive local times, so DST ambiguity cannot
    # be resolved reliably; expected minutes follow the dataset's naive clock.
    result['expected_minutes'] = ((next_starts - result.index).total_seconds() / 60).astype('int64')
    result['coverage_pct'] = result['observed_minutes'] / result['expected_minutes'] * 100
    result['is_complete'] = result['observed_minutes'].eq(result['expected_minutes'])
    result.loc[result['observed_minutes'].eq(0), ['mean_active_power_kw', 'max_minute_active_power_kw']] = float('nan')
    return result.reset_index()


def main() -> None:
    df = load_featured(INPUT)
    featured_path = BASE / 'household_energy_featured.csv'
    df.to_csv(featured_path, index=False)
    print(f'Featured: {len(df):,} rows -> {featured_path.name}')
    for freq, label in [('h', 'hourly'), ('D', 'daily'), ('MS', 'monthly')]:
        summary = aggregate(df, freq)
        target = BASE / f'household_energy_{label}.csv'
        summary.to_csv(target, index=False)
        print(f'{label.title()}: {len(summary):,} periods; '
              f'{(~summary["is_complete"]).sum():,} incomplete -> {target.name}')
    print('Missing minutes remain gaps. Do not interpret observed_energy_kwh '
          'as full-period energy when coverage_pct < 100.')


if __name__ == '__main__':
    main()
