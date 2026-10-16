"""
Initial data loading and basic cleaning
Project: Household Energy Consumption Analytics Dashboard

Input:
    household_power_consumption.txt

Output:
    household_energy_cleaned.csv
"""

from pathlib import Path
import pandas as pd


# File paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "raw" / "household_power_consumption.txt"
OUTPUT_FILE = ROOT / "data" / "processed" / "household_energy_cleaned.csv"


def load_and_clean_data(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Load the raw UCI household power dataset and perform basic cleaning."""

    # 1. Load the semicolon-separated raw file.
    # Treat common missing-value placeholders as NaN.
    df = pd.read_csv(
        input_file,
        sep=";",
        na_values=["?", "NA", "N/A", "null", "None", ""],
        keep_default_na=True,
        low_memory=False,
    )

    print(f"Raw shape: {df.shape}")
    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    # 2. Rename columns for clarity and consistency.
    rename_map = {
        "Date": "date",
        "Time": "time",
        "Global_active_power": "global_active_power_kw",
        "Global_reactive_power": "global_reactive_power_kw",
        "Voltage": "voltage_v",
        "Global_intensity": "global_intensity_a",
        "Sub_metering_1": "sub_metering_1_wh",
        "Sub_metering_2": "sub_metering_2_wh",
        "Sub_metering_3": "sub_metering_3_wh",
    }
    df = df.rename(columns=rename_map)

    # 3. Combine date and time into one proper datetime column.
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    # 4. Convert all measurement columns to numeric.
    measurement_cols = [
        "global_active_power_kw",
        "global_reactive_power_kw",
        "voltage_v",
        "global_intensity_a",
        "sub_metering_1_wh",
        "sub_metering_2_wh",
        "sub_metering_3_wh",
    ]

    for col in measurement_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Remove rows with an invalid timestamp, if any.
    invalid_timestamps = df["timestamp"].isna().sum()
    if invalid_timestamps:
        df = df.dropna(subset=["timestamp"])
    print(f"\nInvalid timestamp rows removed: {invalid_timestamps}")

    # 6. Remove duplicate timestamps, keeping the first occurrence.
    duplicate_timestamps = df.duplicated(subset=["timestamp"]).sum()
    if duplicate_timestamps:
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
    print(f"Duplicate timestamp rows removed: {duplicate_timestamps}")

    # 7. Handle missing electrical measurements.
    # We do NOT replace missing consumption with zero.
    # For this initial clean dataset, rows with incomplete measurement data are removed.
    
    # Missing-value strategy:
    # 25,979 rows (~1.25% of the original dataset) contain missing
    # electrical measurements. These rows are removed rather than imputed.
    # This preserves observed measurement values but introduces gaps in the
    # otherwise one-minute time series. These gaps should be considered
    # during aggregation and time-series analysis.
    
    missing_measurement_rows = df[measurement_cols].isna().any(axis=1).sum()
    df = df.dropna(subset=measurement_cols)
    print(f"Rows with missing measurement values removed: {missing_measurement_rows}")

    # 8. Sort chronologically and remove the original date/time text columns.
    df = (
        df.sort_values("timestamp")
          .drop(columns=["date", "time"])
          .reset_index(drop=True)
    )

    # Put timestamp first.
    df = df[["timestamp"] + measurement_cols]

    print(f"\nCleaned shape: {df.shape}")
    print("\nFinal data types:")
    print(df.dtypes)

    print("\nRemaining missing values:")
    print(df.isna().sum())

    return df


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find '{INPUT_FILE}'. "
            "Place the raw UCI dataset in data/raw/ (see README)."
        )

    cleaned_df = load_and_clean_data(INPUT_FILE)

    # 9. Save the cleaned intermediate dataset.
    cleaned_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCleaned dataset saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
