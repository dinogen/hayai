import unittest

import pandas as pd

from app.jobs.diagnostic import _max_drawdown, _realized_pnl_summary, _reconstruct_portfolio, _regression_metrics, _signal_bins


class DiagnosticFunctionsTest(unittest.TestCase):
    def test_regression_metrics_include_directional_and_error_measures(self):
        metrics = _regression_metrics(pd.Series([1.0, -1.0]), pd.Series([1.0, -1.0]))

        self.assertEqual(metrics["n"], 2)
        self.assertAlmostEqual(metrics["pearson"], 1.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["hit_rate"], 1.0)

    def test_max_drawdown_is_negative_after_loss(self):
        self.assertLess(_max_drawdown(pd.Series([0.1, -0.2])), 0.0)

    def test_signal_bins_keep_empty_ranges(self):
        frame = pd.DataFrame({
            "quant_signal": [-2.5, 0.5, 2.5],
            "actual_return_5d": [0.1, -0.1, 0.2],
        })

        bins = _signal_bins(frame)

        self.assertEqual(len(bins), 6)
        self.assertEqual(int(bins.loc[0, "n"]), 1)
        self.assertEqual(int(bins.loc[3, "n"]), 1)
        self.assertEqual(int(bins.loc[5, "n"]), 1)

    def test_reconstruction_handles_short_cash_flow_and_nav(self):
        execution = {
            "initial_capital": 1000.0,
            "trades": pd.DataFrame([
                {"trade_date": "2026-01-02", "instrument_id": 1, "symbol": "AAA", "side": "short", "qty": 2, "price": 100.0, "amount": 200.0},
                {"trade_date": "2026-01-05", "instrument_id": 1, "symbol": "AAA", "side": "cover", "qty": 2, "price": 90.0, "amount": -180.0},
            ]),
            "prices": pd.DataFrame([
                {"trade_date": "2026-01-02", "instrument_id": 1, "symbol": "AAA", "close": 100.0},
                {"trade_date": "2026-01-05", "instrument_id": 1, "symbol": "AAA", "close": 90.0},
            ]),
            "cash": pd.DataFrame(),
            "positions": pd.DataFrame(),
        }

        result = _reconstruct_portfolio(execution)

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["trade_count"], 2)
        self.assertAlmostEqual(result["final_nav_reconstructed"], 1020.0)

    def test_realized_pnl_pairs_long_and_short_closures(self):
        reconstruction = {
            "trades": pd.DataFrame([
                {"symbol": "LONG", "side": "sell", "amount": 110.0, "realized_pnl": 10.0},
                {"symbol": "SHORT", "side": "cover", "amount": -90.0, "realized_pnl": 10.0},
            ])
        }

        result = _realized_pnl_summary(reconstruction)

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["closed_trades"], 2)
        self.assertAlmostEqual(result["realized_pnl"], 20.0)

    def test_train_only_preprocessing_uses_train_bounds(self):
        from app.jobs.dataset_builder import _apply_train_only_preprocessing

        frame = pd.DataFrame({
            "trade_date": pd.to_datetime([
                "2024-01-01", "2024-01-02", "2024-01-03",
                "2024-01-04", "2024-01-05", "2024-01-06"
            ]),
            "feature_a": [0.0, 1.0, 2.0, 100.0, 200.0, 300.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        })

        processed = _apply_train_only_preprocessing(frame, ["feature_a", "feature_b"], frame["trade_date"] <= "2024-01-03")

        self.assertAlmostEqual(processed.loc[processed["trade_date"] == "2024-01-05", "feature_a"].iloc[0], 1.99)
        self.assertAlmostEqual(processed.loc[processed["trade_date"] == "2024-01-06", "feature_b"].iloc[0], 29.9)
        self.assertLess(processed["feature_a"].max(), 100.0)


if __name__ == "__main__":
    unittest.main()
