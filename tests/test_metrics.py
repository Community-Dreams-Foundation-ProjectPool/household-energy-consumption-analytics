"""Task 9: metric calculations, coverage, and malformed-input checks."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energy_metrics import calculate_energy_metrics, compute_hourly_anomalies, validate_aggregates
from household_energy_aggregation import aggregate


class TestEnergyMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        timestamps = pd.date_range("2024-01-27", periods=12 * 24 * 60, freq="min")
        timestamps = timestamps.delete(125)  # Missing minute: never zero-fill
        df = pd.DataFrame({
            "timestamp": timestamps,
            "energy_kwh": 1.5 / 60,
            "global_active_power_kw": 1.5,
            "sub_metering_1_wh": 2.0,
            "sub_metering_2_wh": 3.0,
            "sub_metering_3_wh": 4.0,
        })
        cls.data = {
            "hourly": aggregate(df, "h"),
            "daily": aggregate(df, "D"),
            "monthly": aggregate(df, "MS"),
        }

    def test_all_eight_metrics_and_coverage(self):
        m = calculate_energy_metrics(self.data)
        s = m["summary"]
        expected_n = 12 * 24 * 60 - 1
        self.assertEqual(s["observed_minutes"], expected_n)
        self.assertEqual(s["expected_minutes"], expected_n + 1)
        self.assertAlmostEqual(s["observed_energy_kwh"], expected_n * 1.5 / 60)
        self.assertAlmostEqual(s["mean_active_power_kw"], 1.5)
        self.assertAlmostEqual(s["peak_hour_mean_kw"], 1.5)
        self.assertEqual(s["incomplete_days"], 1)
        self.assertEqual(len(m["hourly_profile"]), 24)
        self.assertTrue(np.allclose(m["hourly_profile"]["mean_active_power_kw"], 1.5))
        self.assertEqual(len(m["monthly_trend"]), 2)
        self.assertEqual(len(m["weekday_weekend"]), 2)
        self.assertTrue(np.allclose(m["weekday_weekend"]["mean_active_power_kw"], 1.5))
        self.assertAlmostEqual(m["submetering"]["share_pct"].sum(), 100.0)
        self.assertEqual(s["high_anomaly_hours"], 0)
        self.assertEqual(s["low_anomaly_hours"], 0)

    def test_detect_hourly_anomaly(self):
        hourly = self.data["hourly"].copy()
        target = hourly["period_start"].dt.hour.eq(12)
        i = hourly[target].index[-1]
        hourly.loc[i, "mean_active_power_kw"] = 8.0
        flagged = compute_hourly_anomalies(hourly)
        self.assertEqual(flagged.loc[flagged["period_start"].eq(hourly.loc[i, "period_start"]),
                                     "anomaly"].iloc[0], "high")

    def test_reject_missing_columns(self):
        broken = dict(self.data)
        broken["hourly"] = broken["hourly"].drop(columns=["observed_minutes"])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_aggregates(broken)

    def test_reject_inconsistent_totals(self):
        broken = {name: frame.copy() for name, frame in self.data.items()}
        broken["monthly"].loc[0, "observed_energy_kwh"] += 100
        with self.assertRaisesRegex(ValueError, "totals do not match"):
            validate_aggregates(broken)

    def test_reject_incorrect_coverage(self):
        broken = {name: frame.copy() for name, frame in self.data.items()}
        broken["daily"].loc[0, "coverage_pct"] = 100
        with self.assertRaisesRegex(ValueError, "coverage percentages"):
            validate_aggregates(broken)


if __name__ == "__main__":
    unittest.main()
