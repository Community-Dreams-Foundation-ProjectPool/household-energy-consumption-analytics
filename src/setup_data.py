"""First-run setup for the Household Energy Dashboard.

Downloads the original UCI dataset if needed, then runs the original
Task 4/6/7 scripts in order. Does not replace missing readings with zeros.

The package includes real Task 7 charts, so the Streamlit app remains
usable in chart-only mode when the download is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent      # the src/ folder
ROOT = BASE.parent                          # repository root
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW = RAW_DIR / "household_power_consumption.txt"
DOWNLOAD = RAW_DIR / "uci_household_download.zip"
REQUIRED_OUTPUTS = [
    PROCESSED / "household_energy_cleaned.csv",
    PROCESSED / "household_energy_featured.csv",
    PROCESSED / "household_energy_hourly.csv",
    PROCESSED / "household_energy_daily.csv",
    PROCESSED / "household_energy_monthly.csv",
]
CHARTS = list((ROOT / "plots").glob("*.png"))
URLS = [
    "https://archive.ics.uci.edu/static/public/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip",
]


def outputs_ready() -> bool:
    return all(path.is_file() and path.stat().st_size > 100 for path in REQUIRED_OUTPUTS)


def download_archive() -> Path:
    """Fetch a ZIP from UCI, with basic validation and an alternate UCI URL."""
    if DOWNLOAD.exists() and zipfile.is_zipfile(DOWNLOAD):
        return DOWNLOAD
    last_error: Exception | None = None
    for url in URLS:
        partial = DOWNLOAD.with_suffix(".part")
        try:
            print(f"Downloading UCI source dataset: {url}", flush=True)
            request = urllib.request.Request(
                url, headers={"User-Agent": "HouseholdEnergyDashboard/1.0 (educational project)"}
            )
            with urllib.request.urlopen(request, timeout=35) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if not zipfile.is_zipfile(partial):
                raise ValueError("UCI response is not a valid ZIP archive")
            partial.replace(DOWNLOAD)
            print(f"Downloaded {DOWNLOAD.stat().st_size / (1024 * 1024):.1f} MB.")
            return DOWNLOAD
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            print(f"Download attempt failed: {exc}", flush=True)
    raise RuntimeError(f"Could not download the UCI archive: {last_error}")


def extract_text_from_zip(archive: Path, output: Path) -> None:
    """Extract only the known dataset file, including from nested ZIPs.

    Avoid extracting arbitrary paths from untrusted ZIP archives.
    """
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        matches = [n for n in names if Path(n).name.lower() == "household_power_consumption.txt"]
        if matches:
            info = z.getinfo(matches[0])
            if info.file_size > 180 * 1024 * 1024:
                raise ValueError("Raw dataset is unexpectedly large")
            temporary = output.with_suffix(".part")
            try:
                with z.open(info) as source, temporary.open("wb") as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)
                if temporary.stat().st_size < 1_000_000:
                    raise ValueError("Raw dataset is unexpectedly small")
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
            return
        # Some UCI archive distributions contain an inner ZIP.
        inner_zips = [n for n in names if n.lower().endswith(".zip")]
        for name in inner_zips:
            inner = archive.with_name("uci_nested_archive.zip")
            try:
                with z.open(name) as source, inner.open("wb") as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)
                extract_text_from_zip(inner, output)
                return
            except (KeyError, ValueError, zipfile.BadZipFile):
                pass
            finally:
                inner.unlink(missing_ok=True)
    raise FileNotFoundError("No household_power_consumption.txt found in UCI ZIP")


def run_script(name: str) -> None:
    print(f"\nRunning {name} ...", flush=True)
    subprocess.run([sys.executable, str(BASE / name)], cwd=ROOT, check=True)


def main() -> int:
    if outputs_ready():
        print("All five cleaned/featured/aggregate CSVs are already present. Skipping setup.")
        return 0
    if not RAW.exists():
        try:
            extract_text_from_zip(download_archive(), RAW)
            print(f"Extracted {RAW.name} ({RAW.stat().st_size / 1e6:.1f} MB).")
        except Exception as exc:
            print(f"\nCould not prepare the full UCI dataset: {exc}")
            print("Bundled real-data visualizations remain available in the app.")
            print("To retry, run python src/setup_data.py again")
            return 1
    try:
        if not (PROCESSED / "household_energy_cleaned.csv").exists():
            run_script("household_energy_initial_cleaning.py")
        # The aggregation script creates all four analysis CSVs together.
        if not outputs_ready():
            run_script("household_energy_aggregation.py")
        if not outputs_ready():
            raise RuntimeError("One or more expected CSV outputs were not created")
        run_script("household_energy_visualization.py")
    except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
        print(f"\nData preparation failed: {exc}")
        print("The app can still display the six bundled Task 7 plots.")
        return 1
    print("\nSuccess: all analysis CSVs and six visualizations are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
