import unittest
from io import StringIO

import numpy as np

from forecasting import (
    FORECAST_STEPS,
    LOOKBACK_STEPS,
    load_household_data,
    make_train_validation_sequences,
    forecast_next_hour,
)


class LoadHouseholdDataTests(unittest.TestCase):
    def test_parses_uci_rows_and_only_interpolates_short_gaps(self):
        content = StringIO(
            "Date;Time;Global_active_power\n"
            "16/12/2006;17:00:00;1.0\n"
            "16/12/2006;17:01:00;?\n"
            "16/12/2006;17:02:00;3.0\n"
            "16/12/2006;17:15:00;2.0\n"
        )

        result = load_household_data(content)

        self.assertEqual(result.name, "active_power_kw")
        self.assertAlmostEqual(result.loc["2006-12-16 17:00"], 2.0)
        self.assertTrue(np.isfinite(result.loc["2006-12-16 17:15"]))


class SequenceTests(unittest.TestCase):
    def test_sequences_are_chronological_and_ignore_windows_over_long_gaps(self):
        values = np.arange(500, dtype=np.float32)
        values[260:265] = np.nan

        x_train, y_train, x_val, y_val, mean, scale = make_train_validation_sequences(
            values,
            lookback=24,
            horizon=4,
        )

        self.assertEqual(x_train.shape[1], 24)
        self.assertEqual(y_train.shape[1], 4)
        self.assertEqual(x_val.shape[1], 24)
        self.assertEqual(y_val.shape[1], 4)
        self.assertTrue(np.isfinite(x_train).all())
        self.assertTrue(np.isfinite(y_train).all())
        self.assertTrue(np.isfinite(x_val).all())
        self.assertTrue(np.isfinite(y_val).all())
        self.assertAlmostEqual(mean, float(np.nanmean(values[:400])))
        self.assertAlmostEqual(scale, float(np.nanstd(values[:400])), delta=1e-5)
        self.assertGreater(len(x_train), 0)
        self.assertGreater(len(x_val), 0)

    def test_rejects_insufficient_history(self):
        with self.assertRaises(ValueError):
            make_train_validation_sequences(
                np.arange(LOOKBACK_STEPS + FORECAST_STEPS, dtype=np.float32)
            )


class ForecastTests(unittest.TestCase):
    def test_converts_model_output_to_nonnegative_kw(self):
        class FixedModel:
            def predict(self, values, verbose=0):
                return np.array([[-2.0, 0.0, 1.0, 2.0]], dtype=np.float32)

        result = forecast_next_hour(
            FixedModel(),
            np.full(LOOKBACK_STEPS, 2.0),
            mean=2.0,
            scale=0.5,
        )

        np.testing.assert_allclose(result, [1.0, 2.0, 2.5, 3.0])
        self.assertEqual(len(result), FORECAST_STEPS)

    def test_rejects_incomplete_forecast_history(self):
        class UnusedModel:
            pass

        with self.assertRaisesRegex(ValueError, "long data gap"):
            forecast_next_hour(
                UnusedModel(),
                np.full(LOOKBACK_STEPS, np.nan),
                mean=0.0,
                scale=1.0,
            )


if __name__ == "__main__":
    unittest.main()
