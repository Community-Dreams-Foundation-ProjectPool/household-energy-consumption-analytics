"""Task 10 filtering, selected-month totals, heatmap and metric regression."""
from datetime import date
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dashboard_filters import filter_aggregates, hourly_weekday_heatmap, daily_weekday_profile
from household_energy_aggregation import aggregate
from energy_metrics import calculate_energy_metrics, validate_aggregates


class TestDashboardFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        timestamps = pd.date_range("2024-01-27", periods=18 * 24 * 60, freq="min")
        timestamps = timestamps.delete(140)  # Missing observation on Jan 27
        power = 1 + timestamps.dayofweek.to_numpy() * 0.2
        df = pd.DataFrame({
            "timestamp": timestamps,
            "energy_kwh": power / 60,
            "global_active_power_kw": power,
            "sub_metering_1_wh": 2.0,
            "sub_metering_2_wh": 3.0,
            "sub_metering_3_wh": 4.0,
        })
        cls.data = {name: aggregate(df, freq) for name, freq in
                    [("hourly", "h"), ("daily", "D"), ("monthly", "MS")]}
        validate_aggregates(cls.data)

    def test_date_filter_includes_both_endpoints(self):
        selected = filter_aggregates(self.data, date(2024, 1, 30), date(2024, 2, 2))
        self.assertEqual(len(selected["daily"]), 4)
        self.assertEqual(len(selected["hourly"]), 4 * 24)
        self.assertEqual(len(selected["monthly"]), 2)
        self.assertEqual(selected["monthly"]["expected_minutes"].sum(), 4 * 1440)
        self.assertAlmostEqual(selected["monthly"]["observed_energy_kwh"].sum(),
                               selected["hourly"]["observed_energy_kwh"].sum())
        self.assertEqual(calculate_energy_metrics(selected)["summary"]["observed_minutes"], 4 * 1440)

    def test_weekend_filter_rebuilds_monthly_not_full_month(self):
        selected = filter_aggregates(self.data, date(2024, 1, 27), date(2024, 2, 13),
                                     day_type="Weekends")
        self.assertTrue(selected["daily"]["period_start"].dt.dayofweek.ge(5).all())
        self.assertEqual(len(selected["daily"]), 6)
        self.assertEqual(selected["monthly"]["expected_minutes"].sum(), 6 * 1440)
        self.assertEqual(calculate_energy_metrics(selected)["summary"]["incomplete_days"], 1)

    def test_season_month_year_combination(self):
        selected = filter_aggregates(self.data, date(2024, 1, 27), date(2024, 2, 13),
                                     year=2024, season="Winter", month=2)
        self.assertEqual(len(selected["daily"]), 13)
        self.assertEqual(len(selected["monthly"]), 1)
        self.assertTrue(selected["daily"]["period_start"].dt.month.eq(2).all())

    def test_empty_selection_and_bad_date(self):
        self.assertIsNone(filter_aggregates(self.data, date(2024, 1, 27), date(2024, 2, 13),
                                            season="Summer"))
        with self.assertRaisesRegex(ValueError, "Start date"):
            filter_aggregates(self.data, date(2024, 2, 2), date(2024, 1, 1))

    def test_weekday_profile_and_heatmap(self):
        hm = hourly_weekday_heatmap(self.data["hourly"])
        self.assertEqual(hm.shape, (7, 24))
        self.assertAlmostEqual(hm.loc[0, 8], 1.0)
        self.assertAlmostEqual(hm.loc[6, 8], 2.2)
        prof = daily_weekday_profile(self.data["daily"])
        self.assertEqual(len(prof), 7)
        self.assertAlmostEqual(prof.loc[0, "mean_active_power_kw"], 1.0)
        self.assertAlmostEqual(prof.loc[6, "mean_active_power_kw"], 2.2)


if __name__ == "__main__":
    unittest.main()


class TestMinutePreview(unittest.TestCase):
    def test_preview_only_matching_dates_and_bounded_rows(self):
        from tempfile import TemporaryDirectory
        from dashboard_filters import read_cleaned_minute_preview
        with TemporaryDirectory() as folder:
            path = Path(folder) / "cleaned.csv"
            pd.DataFrame({
                "timestamp": ["2024-01-01 00:00", "2024-01-02 00:00",
                              "2024-01-02 00:01", "2024-01-03 00:00"],
                "global_active_power_kw": [1.0, 2.0, 3.0, 4.0],
            }).to_csv(path, index=False)
            preview = read_cleaned_minute_preview(str(path), (date(2024, 1, 2),), limit=1)
            self.assertEqual(len(preview), 1)
            self.assertEqual(preview.loc[0, "global_active_power_kw"], 2.0)
            self.assertTrue(read_cleaned_minute_preview(str(path), (date(2024, 1, 4),)).empty)
