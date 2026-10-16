"""Quick synthetic smoke tests. No download or full dataset required."""
import sys
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from household_energy_aggregation import aggregate, season_from_month
from household_energy_visualization import (
    plot_daily_energy_trend, plot_monthly_avg_power_trend,
    plot_avg_by_day_of_week, plot_hour_by_weekday_heatmap,
    plot_submetering_share, plot_anomalous_hours_by_month,
    compute_submetering_shares,
)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        times = pd.date_range("2024-01-25", periods=12 * 24 * 60, freq="min")
        times = times.delete(75)  # deliberately missing minute
        t = pd.Series(times)
        power = 1.5 + 0.4 * np.sin(2 * np.pi * t.dt.hour.to_numpy() / 24)
        cls.df = pd.DataFrame({
            "timestamp": times,
            "global_active_power_kw": power,
            "energy_kwh": power / 60,
            "sub_metering_1_wh": 2.0,
            "sub_metering_2_wh": 3.0,
            "sub_metering_3_wh": 4.0,
            "day_name": t.dt.day_name(),
            "hour": t.dt.hour,
        })
        cls.hourly = aggregate(cls.df, "h")
        cls.daily = aggregate(cls.df, "D")
        cls.monthly = aggregate(cls.df, "MS")

    def test_coverage_and_energy(self):
        self.assertEqual(int(self.hourly["observed_minutes"].sum()), len(self.df))
        self.assertEqual(int(self.hourly["expected_minutes"].sum()), 12 * 24 * 60)
        self.assertEqual(int((~self.hourly["is_complete"]).sum()), 1)
        self.assertAlmostEqual(self.daily["observed_energy_kwh"].sum(),
                               self.hourly["observed_energy_kwh"].sum())
        self.assertAlmostEqual(self.monthly["observed_energy_kwh"].sum(),
                               self.hourly["observed_energy_kwh"].sum())

    def test_seasons(self):
        self.assertEqual(list(season_from_month(pd.Series([1, 4, 7, 10]))),
                         ["Winter", "Spring", "Summer", "Fall"])

    def test_submetering(self):
        shares = compute_submetering_shares(self.monthly)
        cols = ["Kitchen", "Laundry", "Water Heater / AC", "Other (unmetered)"]
        self.assertTrue(np.allclose(shares[cols].sum(axis=1), 100))

    def test_all_six_plots(self):
        for fn, data in [
            (plot_daily_energy_trend, self.daily),
            (plot_monthly_avg_power_trend, self.monthly),
            (plot_avg_by_day_of_week, self.df),
            (plot_hour_by_weekday_heatmap, self.df),
            (plot_submetering_share, self.monthly),
            (plot_anomalous_hours_by_month, self.hourly),
        ]:
            with self.subTest(plot=fn.__name__):
                fig = fn(data)
                self.assertGreater(len(fig.axes), 0)
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
