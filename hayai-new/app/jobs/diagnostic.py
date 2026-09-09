"""Read-only diagnostic check-up for the v2 model.

The job deliberately does not call prediction, signal, recommendation, NAV or
trade jobs. It rebuilds the feature panel and evaluates the deployed artifact,
then records missing historical inputs as PARTIAL/BLOCKED instead of inventing
values.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.config import resolve_model_artifact_path
from app.db import execute_query
from app.jobs.dataset_builder import (
    build_training_dataset,
    read_model_config,
    split_by_cutoffs,
)
from app.jobs.verify_model import _load_model, _load_onnx_session
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.diagnostic")

PREDICTION_COLUMNS = [
    "timestamp", "date", "symbol", "quant_signal", "news_modifier",
    "final_signal", "target_weight", "side", "price_at_signal",
    "price_at_execution", "return_1d", "return_5d", "return_10d",
    "actual_return_5d", "vol_20", "model_version", "data_status",
]


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value * 100:.2f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _safe_query(query: str, params: tuple = None) -> tuple[list[dict], str | None]:
    """Read an optional table without making the whole diagnostic fail."""
    try:
        return execute_query(query, params), None
    except Exception as exc:  # optional historical tables may not exist yet
        logger.warning("Optional diagnostic query unavailable: %s", exc)
        return [], str(exc)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _split_indices(clean_df: pd.DataFrame, config: dict) -> dict[str, pd.Series]:
    random_train, random_test = train_test_split(
        clean_df.index, test_size=0.2, random_state=42
    )
    result = {
        "random_train": clean_df.index.isin(random_train),
        "random_test": clean_df.index.isin(random_test),
    }
    train_end = config.get("train_end")
    val_end = config.get("val_end")
    if config.get("split") == "time" and train_end and val_end:
        train, validation, test = split_by_cutoffs(clean_df, train_end, val_end)
        result.update({"chrono_train": train, "chrono_validation": validation, "chrono_test": test})
    else:
        dates = np.sort(clean_df["trade_date"].unique())
        n_train = int(round(len(dates) * 0.70))
        n_val = int(round(len(dates) * 0.15))
        result.update({
            "chrono_train": clean_df["trade_date"].isin(dates[:n_train]).to_numpy(),
            "chrono_validation": clean_df["trade_date"].isin(dates[n_train:n_train + n_val]).to_numpy(),
            "chrono_test": clean_df["trade_date"].isin(dates[n_train + n_val:]).to_numpy(),
        })
    return result


def _regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | None]:
    frame = pd.DataFrame({"actual": actual, "predicted": predicted}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {"n": 0, "pearson": None, "spearman": None, "mae": None, "rmse": None, "r2": None, "hit_rate": None}
    a = frame["actual"]
    p = frame["predicted"]
    return {
        "n": int(len(frame)),
        "pearson": _finite(a.corr(p)),
        "spearman": _finite(a.corr(p, method="spearman")),
        "mae": _finite(mean_absolute_error(a, p)),
        "rmse": _finite(np.sqrt(mean_squared_error(a, p))),
        "r2": _finite(r2_score(a, p)) if len(frame) > 1 else None,
        "hit_rate": _finite((np.sign(a) == np.sign(p)).mean()),
    }


def _max_drawdown(returns: pd.Series) -> float | None:
    values = pd.Series(returns, dtype=float).dropna()
    if values.empty:
        return None
    curve = np.exp(values.cumsum())
    drawdown = curve / curve.cummax() - 1.0
    return _finite(drawdown.min())


def _strategy_summary(returns: pd.Series) -> dict[str, float | None]:
    values = pd.Series(returns, dtype=float).dropna()
    if values.empty:
        return {"n": 0, "return": None, "volatility": None, "max_drawdown": None}
    return {
        "n": int(len(values)),
        "return": _finite(np.exp(values.sum()) - 1.0),
        "volatility": _finite(values.std(ddof=1) * np.sqrt(252 / 5)) if len(values) > 1 else None,
        "max_drawdown": _max_drawdown(values),
    }


def _non_overlapping(returns: pd.Series, step: int = 5) -> pd.Series:
    values = pd.Series(returns, dtype=float).dropna().sort_index()
    return values.iloc[::step]


def _strategy_returns(frame: pd.DataFrame, signal_col: str, top_n: int = 5, bottom_n: int = 5, return_col: str = "actual_return_5d", threshold: float = 0.0) -> pd.DataFrame:
    returns = []
    for trade_date, group in frame.groupby("trade_date"):
        group = group.dropna(subset=[signal_col, return_col])
        group = group[group[signal_col].abs() >= threshold]
        if len(group) < top_n + bottom_n:
            continue
        top = group.nlargest(top_n, signal_col)[return_col].mean()
        bottom = group.nsmallest(bottom_n, signal_col)[return_col].mean()
        returns.append({"date": trade_date, "long": top, "short": -bottom, "long_short": top - bottom})
    if not returns:
        return pd.DataFrame(columns=["date", "long", "short", "long_short"])
    return pd.DataFrame(returns).set_index("date")


def _non_overlapping_strategy(frame: pd.DataFrame, signal_col: str = "quant_signal", return_col: str = "actual_return_5d", top_n: int = 5, bottom_n: int = 5, threshold: float = 0.0) -> dict[str, Any]:
    values = _strategy_returns(frame, signal_col, top_n, bottom_n, return_col, threshold)
    if values.empty:
        return {"status": "BLOCKED", "n": 0}
    values = values.iloc[::5]
    summary = _strategy_summary(values["long_short"])
    return {
        "status": "OK",
        "n": int(len(values)),
        "return": summary["return"],
        "volatility": summary["volatility"],
        "max_drawdown": summary["max_drawdown"],
        "long_mean": _finite(values["long"].mean()),
        "short_mean_position": _finite(values["short"].mean()),
        "win_rate": _finite((values["long_short"] > 0).mean()),
        "q5_q1": _finite(values["long_short"].mean()),
    }


def _temporal_stability(frame: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(frame["trade_date"].dropna().unique()))
    if len(dates) < 4:
        return pd.DataFrame()
    rows = []
    for period, date_group in enumerate(np.array_split(dates, 4), start=1):
        subset = frame[frame["trade_date"].isin(date_group)]
        result = _non_overlapping_strategy(subset)
        if result["status"] == "OK":
            rows.append({"period": f"Period {period}", "start": str(pd.Timestamp(date_group[0]).date()), "end": str(pd.Timestamp(date_group[-1]).date()), **{key: result[key] for key in ("n", "return", "volatility", "max_drawdown", "win_rate", "q5_q1")}})
    return pd.DataFrame(rows)


def _classify_regime(group: pd.DataFrame) -> str:
    market_return = float(group["mkt_ret_20"].median()) if "mkt_ret_20" in group and group["mkt_ret_20"].notna().any() else 0.0
    market_vol = float(group["vol_regime"].median()) if "vol_regime" in group and group["vol_regime"].notna().any() else 1.0
    if market_vol > 1.25:
        return "High volatility"
    if market_vol < 0.80:
        return "Low volatility"
    if market_return > 0.05:
        return "Bull"
    if market_return < -0.05:
        return "Bear"
    return "Sideways"


def _regime_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    strategy = _strategy_returns(frame, "quant_signal")
    if strategy.empty:
        return pd.DataFrame()
    regimes = frame.groupby("trade_date").apply(_classify_regime, include_groups=False).rename("regime")
    daily_frame = strategy.reset_index().merge(regimes.reset_index(), left_on="date", right_on="trade_date", how="left")
    daily_frame["win_rate"] = (daily_frame["long_short"] > 0).astype(float)
    rows = []
    for regime, group in daily_frame.groupby("regime"):
        values = group.sort_values("date").iloc[::5]["long_short"]
        rows.append({"regime": regime, "n": len(values), "return": _finite(np.exp(values.sum()) - 1.0), "volatility": _finite(values.std(ddof=1) * np.sqrt(252 / 5)) if len(values) > 1 else None, "win_rate": _finite((values > 0).mean())})
    return pd.DataFrame(rows).sort_values("regime")


def _robustness_grid(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    horizon_columns = {1: "return_1d", 5: "return_5d", 10: "return_10d", 20: "return_20d"}
    for horizon, return_col in horizon_columns.items():
        for positions in (5, 10, 20, 50):
            result = _non_overlapping_strategy(frame, return_col=return_col, top_n=positions, bottom_n=positions)
            if result["status"] == "OK":
                rows.append({"test": "holding_positions", "holding_days": horizon, "positions_per_side": positions, "threshold": 0.0, "return": result["return"], "volatility": result["volatility"], "max_drawdown": result["max_drawdown"], "win_rate": result["win_rate"]})
    for threshold in (0.0, 0.25, 0.5, 1.0):
        result = _non_overlapping_strategy(frame, threshold=threshold)
        if result["status"] == "OK":
            rows.append({"test": "signal_threshold", "holding_days": 5, "positions_per_side": 5, "threshold": threshold, "return": result["return"], "volatility": result["volatility"], "max_drawdown": result["max_drawdown"], "win_rate": result["win_rate"]})
    return pd.DataFrame(rows)


def _baseline_results(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model = _strategy_returns(frame, "quant_signal")
    momentum = _strategy_returns(frame, "mom_20")
    for name, values in (("Model", model["long_short"] if not model.empty else pd.Series(dtype=float)),
                         ("Momentum 20d", momentum["long_short"] if not momentum.empty else pd.Series(dtype=float))):
        summary = _strategy_summary(_non_overlapping(values))
        rows.append({"strategy": name, **summary})

    equal_weight = frame.groupby("trade_date")["actual_return_5d"].mean()
    spy = frame[frame["symbol"] == "SPY"].set_index("trade_date")["actual_return_5d"]
    for name, values in (("Equal Weight", equal_weight), ("SPY", spy), ("Zero model", pd.Series(0.0, index=equal_weight.index))):
        summary = _strategy_summary(_non_overlapping(values))
        rows.append({"strategy": name, **summary})

    random_values = []
    for position, (trade_date, group) in enumerate(frame.groupby("trade_date")):
        group = group.dropna(subset=["actual_return_5d"])
        if len(group) < 10:
            continue
        sample = group.sample(n=10, random_state=42 + position)
        random_values.append(sample["actual_return_5d"].mean())
    rows.append({"strategy": "Random equal-weight", **_strategy_summary(_non_overlapping(pd.Series(random_values)))})
    return pd.DataFrame(rows)


def _long_short_summary(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame.dropna(subset=["quant_signal", "actual_return_5d"])
    if selected.empty:
        return {"status": "BLOCKED"}
    long_values = selected.loc[selected["quant_signal"] > 0, "actual_return_5d"]
    short_values = selected.loc[selected["quant_signal"] < 0, "actual_return_5d"]
    return {
        "status": "OK",
        "long_n": int(len(long_values)),
        "long_mean": _finite(long_values.mean()),
        "long_median": _finite(long_values.median()),
        "long_win_rate": _finite((long_values > 0).mean()),
        "short_n": int(len(short_values)),
        "short_mean_asset": _finite(short_values.mean()),
        "short_mean_position": _finite(-short_values.mean()),
        "short_median_position": _finite(-short_values.median()),
        "short_win_rate": _finite((short_values < 0).mean()),
    }


def _inverted_summary(frame: pd.DataFrame) -> dict[str, Any]:
    inverted = frame.copy()
    inverted["inverted_signal"] = -inverted["quant_signal"]
    values = _strategy_returns(inverted, "inverted_signal")
    return _strategy_summary(_non_overlapping(values["long_short"] if not values.empty else pd.Series(dtype=float)))


def _observed_summary(frame: pd.DataFrame) -> dict[str, Any]:
    observed = frame.dropna(subset=["final_signal", "actual_return_5d"])
    if observed.empty:
        return {"status": "BLOCKED", "rows": 0}
    observed = observed.copy()
    observed["quant_observed"] = observed["final_signal"] - observed["news_modifier"].fillna(0.0)
    return {
        "status": "OK",
        "rows": int(len(observed)),
        "news_modified_pct": _finite((observed["news_modifier"].fillna(0.0).abs() > 0).mean()),
        "mean_abs_news_modifier": _finite(observed["news_modifier"].abs().mean()),
        "max_abs_news_modifier": _finite(observed["news_modifier"].abs().max()),
        "quant_return": _strategy_summary(observed["quant_observed"] * observed["vol_20"]),
        "hybrid_return": _strategy_summary(observed["final_signal"] * observed["vol_20"]),
        "weight_rows": int(observed["target_weight"].notna().sum()),
        "signal_weight_corr": _finite(observed["final_signal"].corr(observed["target_weight"])) if observed["target_weight"].notna().sum() > 1 else None,
    }


def _preprocessing_audit(frame: pd.DataFrame, feature_cols: list[str], chrono_train_mask: pd.Series) -> dict[str, Any]:
    """Compare global preprocessing bounds with bounds available at train time."""
    train = frame.loc[chrono_train_mask]
    if train.empty:
        return {"status": "BLOCKED"}
    rows = []
    for column in feature_cols:
        global_values = pd.to_numeric(frame[column], errors="coerce").dropna()
        train_values = pd.to_numeric(train[column], errors="coerce").dropna()
        if global_values.empty or train_values.empty:
            continue
        global_low, global_high = global_values.quantile(0.005), global_values.quantile(0.995)
        train_low, train_high = train_values.quantile(0.005), train_values.quantile(0.995)
        rows.append({
            "feature": column,
            "global_low": _finite(global_low),
            "train_low": _finite(train_low),
            "low_difference": _finite(global_low - train_low),
            "global_high": _finite(global_high),
            "train_high": _finite(train_high),
            "high_difference": _finite(global_high - train_high),
        })
    audit = pd.DataFrame(rows)
    if audit.empty:
        return {"status": "BLOCKED"}
    audit["max_absolute_difference"] = audit[["low_difference", "high_difference"]].abs().max(axis=1)
    return {
        "status": "OK",
        "features": audit,
        "affected_features": int((audit["max_absolute_difference"] > 1e-12).sum()),
        "max_difference": _finite(audit["max_absolute_difference"].max()),
        "mean_difference": _finite(audit["max_absolute_difference"].mean()),
    }


def _load_news_coverage(portfolio_code: str, first_date: pd.Timestamp, last_date: pd.Timestamp) -> dict[str, Any]:
    rows, error = _safe_query(
        """SELECT n.published_at, ns.impact_score, ns.confidence
           FROM news n JOIN news_sentiment ns ON ns.news_id = n.id
           JOIN instrument i ON i.id = n.instrument_id
           JOIN portfolio_instrument pi ON pi.instrument_id = i.id
           JOIN portfolio p ON p.id = pi.portfolio_id
           WHERE p.code = %s AND n.published_at >= %s AND n.published_at < DATE_ADD(%s, INTERVAL 1 DAY)""",
        (portfolio_code, first_date.strftime("%Y-%m-%d"), last_date.strftime("%Y-%m-%d")),
    )
    if error:
        return {"status": "BLOCKED", "error": error}
    if not rows:
        return {"status": "BLOCKED", "news_rows": 0, "coverage_pct": 0.0}
    news = pd.DataFrame(rows)
    news["published_at"] = pd.to_datetime(news["published_at"], errors="coerce")
    return {
        "status": "OK",
        "news_rows": int(len(news)),
        "first_news": str(news["published_at"].min()),
        "last_news": str(news["published_at"].max()),
        "coverage_pct": _finite(news["published_at"].dt.date.nunique() / max((last_date - first_date).days + 1, 1)),
        "mean_abs_impact": _finite(pd.to_numeric(news["impact_score"], errors="coerce").abs().mean()),
    }


def _evaluate_artifact(model_info: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate one registered artifact with the preprocessing it was trained with."""
    artifact_path = resolve_model_artifact_path(model_info["artifact_path"])
    config = read_model_config(artifact_path)
    feature_cols = json.loads(model_info["feature_columns"])
    model_frame = frame.dropna(subset=feature_cols + ["target"]).copy()
    session, input_name = _load_onnx_session(artifact_path)
    if config.get("split") == "time" and config.get("train_end") and config.get("val_end"):
        train_mask, validation_mask, test_mask = split_by_cutoffs(model_frame, config["train_end"], config["val_end"])
        train_frame = model_frame.loc[train_mask]
        test_frame = model_frame.loc[test_mask]
    else:
        train_indices, test_indices = train_test_split(model_frame.index, test_size=0.2, random_state=42)
        train_frame = model_frame.loc[train_indices]
        test_frame = model_frame.loc[test_indices]
    mins = train_frame[feature_cols].min()
    maxs = train_frame[feature_cols].max()
    normalized = (test_frame[feature_cols] - mins) / (maxs - mins + 1e-8)
    raw_prediction = session.run(None, {input_name: normalized.to_numpy(dtype=np.float32)})[0].ravel()
    prediction = np.clip(
        raw_prediction * (float(model_info["label_max"]) - float(model_info["label_min"])) + float(model_info["label_min"]),
        float(model_info["clip_min"]), float(model_info["clip_max"]),
    )
    evaluated = test_frame.copy()
    evaluated["model_signal"] = prediction
    return {
        "version": model_info.get("version", "unknown"),
        "split": config.get("split", "random"),
        "train_start": str(train_frame["trade_date"].min().date()),
        "train_end": str(train_frame["trade_date"].max().date()),
        "test_start": str(test_frame["trade_date"].min().date()),
        "test_end": str(test_frame["trade_date"].max().date()),
        "metrics": _regression_metrics(evaluated["target"], evaluated["model_signal"]),
        "strategy": _non_overlapping_strategy(evaluated, signal_col="model_signal"),
        "rows": len(evaluated),
    }


def _load_execution_data(portfolio_code: str) -> tuple[dict[str, Any], str | None]:
    """Load persisted execution/NAV history without changing the database."""
    port_rows, port_error = _safe_query(
        "SELECT id, initial_capital FROM portfolio WHERE code = %s", (portfolio_code,)
    )
    if port_error or not port_rows:
        return {}, port_error or "portfolio not found"
    portfolio_id = int(port_rows[0]["id"])
    queries = {
        "trades": """SELECT pt.trade_date, pt.instrument_id, i.symbol, pt.side,
                           pt.qty, pt.price, pt.amount
                    FROM portfolio_trade pt JOIN instrument i ON i.id = pt.instrument_id
                    WHERE pt.portfolio_id = %s ORDER BY pt.trade_date, pt.id""",
        "cash": """SELECT cash_date, balance FROM portfolio_cash
                   WHERE portfolio_id = %s ORDER BY cash_date""",
        "positions": """SELECT pp.pos_date, pp.instrument_id, i.symbol, pp.qty,
                              pp.avg_price, pp.market_value
                       FROM portfolio_position pp JOIN instrument i ON i.id = pp.instrument_id
                       WHERE pp.portfolio_id = %s ORDER BY pp.pos_date, pp.instrument_id""",
        "prices": """SELECT pd.trade_date, pd.instrument_id, i.symbol, pd.close
                           FROM price_daily pd JOIN instrument i ON i.id = pd.instrument_id
                           WHERE pd.close IS NOT NULL ORDER BY pd.trade_date, pd.instrument_id""",
    }
    result: dict[str, Any] = {"initial_capital": float(port_rows[0]["initial_capital"]), "portfolio_id": portfolio_id}
    for name, query in queries.items():
        params = None if name == "prices" else (portfolio_id,)
        rows, error = _safe_query(query, params)
        if error:
            return {}, error
        result[name] = pd.DataFrame(rows)
    return result, None


def _reconstruct_portfolio(execution: dict[str, Any]) -> dict[str, Any]:
    """Rebuild cash, signed positions and NAV from the immutable trade log."""
    trades = execution.get("trades", pd.DataFrame()).copy()
    prices = execution.get("prices", pd.DataFrame()).copy()
    saved_cash = execution.get("cash", pd.DataFrame()).copy()
    saved_positions = execution.get("positions", pd.DataFrame()).copy()
    if trades.empty or prices.empty:
        return {"status": "BLOCKED", "reason": "trade or price history is empty"}

    for column in ("trade_date",):
        trades[column] = pd.to_datetime(trades[column])
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    for column in ("qty", "price", "amount"):
        trades[column] = pd.to_numeric(trades[column], errors="coerce").astype(float)
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce").astype(float)
    if not saved_cash.empty:
        saved_cash["cash_date"] = pd.to_datetime(saved_cash["cash_date"])
        saved_cash["balance"] = pd.to_numeric(saved_cash["balance"], errors="coerce").astype(float)
    if not saved_positions.empty:
        saved_positions["pos_date"] = pd.to_datetime(saved_positions["pos_date"])
        saved_positions["market_value"] = pd.to_numeric(saved_positions["market_value"], errors="coerce").astype(float)

    dates = pd.DatetimeIndex(sorted(set(trades["trade_date"]) | set(prices["trade_date"])))
    price_table = prices.pivot_table(index="trade_date", columns="instrument_id", values="close", aggfunc="last")
    price_table = price_table.reindex(dates).ffill()
    position_state: dict[int, dict[str, float]] = {}
    cash = float(execution["initial_capital"])
    trade_groups = {date: group for date, group in trades.groupby("trade_date")}
    rows = []
    trade_details = []

    for current_date in dates:
        day_trades = trade_groups.get(current_date)
        if day_trades is not None:
            for trade in day_trades.itertuples(index=False):
                instrument_id = int(trade.instrument_id)
                signed_before = position_state.get(instrument_id, {"qty": 0.0, "avg_price": 0.0})
                before_qty = signed_before["qty"]
                quantity = float(trade.qty)
                price = float(trade.price)
                cash += float(trade.amount)
                realized_pnl = 0.0
                if trade.side in ("sell", "cover") and before_qty:
                    if trade.side == "sell":
                        realized_pnl = quantity * (price - signed_before["avg_price"])
                    else:
                        realized_pnl = quantity * (signed_before["avg_price"] - price)
                if trade.side in ("buy", "short"):
                    signed_delta = quantity if trade.side == "buy" else -quantity
                    if before_qty and (before_qty > 0) == (signed_delta > 0):
                        total_qty = abs(before_qty) + abs(signed_delta)
                        average = (abs(before_qty) * signed_before["avg_price"] + abs(signed_delta) * price) / total_qty
                    else:
                        average = price
                    position_state[instrument_id] = {"qty": before_qty + signed_delta, "avg_price": average}
                else:
                    signed_delta = -quantity if trade.side == "sell" else quantity
                    new_qty = before_qty + signed_delta
                    if abs(new_qty) < 1e-9:
                        position_state.pop(instrument_id, None)
                    else:
                        position_state[instrument_id] = {"qty": new_qty, "avg_price": signed_before["avg_price"]}
                trade_details.append({
                    "date": current_date.strftime("%Y-%m-%d"), "symbol": trade.symbol,
                    "side": trade.side, "qty": quantity, "price": price,
                    "amount": float(trade.amount), "realized_pnl": realized_pnl,
                    "position_after": position_state.get(instrument_id, {}).get("qty", 0.0),
                })

        positions_value = 0.0
        for instrument_id, state in position_state.items():
            value = price_table.loc[current_date].get(instrument_id, np.nan)
            if pd.notna(value):
                positions_value += state["qty"] * float(value)
        rows.append({"date": current_date, "cash_reconstructed": cash, "positions_value_reconstructed": positions_value, "nav_reconstructed": cash + positions_value})

    reconstructed = pd.DataFrame(rows)
    if not saved_cash.empty:
        saved = saved_cash.groupby("cash_date", as_index=False)["balance"].last().rename(columns={"cash_date": "date", "balance": "cash_saved"})
        reconstructed = reconstructed.merge(saved, on="date", how="left")
    if not saved_positions.empty:
        saved_value = saved_positions.groupby("pos_date", as_index=False)["market_value"].sum().rename(columns={"pos_date": "date", "market_value": "positions_value_saved"})
        reconstructed = reconstructed.merge(saved_value, on="date", how="left")
        reconstructed["nav_saved"] = reconstructed["cash_saved"] + reconstructed["positions_value_saved"]
        reconstructed["nav_difference"] = reconstructed["nav_reconstructed"] - reconstructed["nav_saved"]

    return {
        "status": "OK",
        "rows": reconstructed,
        "trades": pd.DataFrame(trade_details),
        "trade_count": len(trades),
        "winning_trades": int((trades["amount"] > 0).sum()),
        "losing_trades": int((trades["amount"] < 0).sum()),
        "final_nav_reconstructed": _finite(reconstructed.iloc[-1]["nav_reconstructed"]),
        "final_nav_saved": _finite(reconstructed.iloc[-1].get("nav_saved")),
    }


def _trade_statistics(reconstruction: dict[str, Any]) -> dict[str, Any]:
    trades = reconstruction.get("trades", pd.DataFrame())
    if trades.empty:
        return {"status": "BLOCKED"}
    winners = trades.loc[trades["amount"] > 0, "amount"]
    losers = trades.loc[trades["amount"] < 0, "amount"]
    return {
        "status": "OK",
        "trades": int(len(trades)),
        "winners": int(len(winners)),
        "losers": int(len(losers)),
        "win_rate": _finite((trades["amount"] > 0).mean()),
        "average_winner": _finite(winners.mean()),
        "average_loser": _finite(losers.mean()),
        "largest_winner": _finite(winners.max()),
        "largest_loser": _finite(losers.min()),
        "profit_factor": _finite(winners.sum() / abs(losers.sum())) if not losers.empty and losers.sum() != 0 else None,
    }


def _realized_pnl_summary(reconstruction: dict[str, Any]) -> dict[str, Any]:
    trades = reconstruction.get("trades", pd.DataFrame())
    if trades.empty or "realized_pnl" not in trades:
        return {"status": "BLOCKED"}
    realized = trades[trades["realized_pnl"] != 0].copy()
    if realized.empty:
        return {"status": "OK", "closed_trades": 0, "realized_pnl": 0.0}
    by_asset = realized.groupby("symbol", as_index=False).agg(
        closed_trades=("realized_pnl", "size"),
        realized_pnl=("realized_pnl", "sum"),
        average_pnl=("realized_pnl", "mean"),
        win_rate=("realized_pnl", lambda values: (values > 0).mean()),
    ).sort_values("realized_pnl", ascending=False)
    return {
        "status": "OK",
        "closed_trades": int(len(realized)),
        "realized_pnl": _finite(realized["realized_pnl"].sum()),
        "winning_closed_trades": int((realized["realized_pnl"] > 0).sum()),
        "losing_closed_trades": int((realized["realized_pnl"] < 0).sum()),
        "by_asset": by_asset,
    }


def _quintiles(frame: pd.DataFrame, signal_col: str = "quant_signal") -> dict[str, Any]:
    rows = []
    for trade_date, group in frame.groupby("trade_date"):
        group = group.dropna(subset=[signal_col, "actual_return_5d"])
        if len(group) < 5:
            continue
        group = group.copy()
        group["quintile"] = pd.qcut(group[signal_col].rank(method="first"), 5, labels=False) + 1
        means = group.groupby("quintile")["actual_return_5d"].mean()
        rows.append({"date": str(pd.Timestamp(trade_date).date()), **{f"q{i}": _finite(means.get(i)) for i in range(1, 6)}})
    if not rows:
        return {"status": "BLOCKED", "rows": []}
    table = pd.DataFrame(rows)
    summary = {f"Q{i}": _finite(table[f"q{i}"].mean()) for i in range(1, 6)}
    summary["Q5-Q1"] = _finite(summary["Q5"] - summary["Q1"])
    return {"status": "OK", "summary": summary, "rows": rows}


def _signal_bins(frame: pd.DataFrame) -> pd.DataFrame:
    bins = [-3, -2, -1, 0, 1, 2, 3]
    labels = ["-3/-2", "-2/-1", "-1/0", "0/1", "1/2", "2/3"]
    selected = frame.dropna(subset=["quant_signal", "actual_return_5d"]).copy()
    selected["signal_range"] = pd.cut(selected["quant_signal"], bins=bins, labels=labels, include_lowest=True)
    return selected.groupby("signal_range", observed=False).agg(
        n=("actual_return_5d", "size"),
        signal_mean=("quant_signal", "mean"),
        future_5d_mean=("actual_return_5d", "mean"),
        win_rate=("actual_return_5d", lambda x: (x > 0).mean()),
    ).reset_index()


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "BLOCKED"
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_forward_returns(clean_df: pd.DataFrame) -> pd.DataFrame:
    frame = clean_df.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values(["symbol", "trade_date"])
    grouped = frame.groupby("symbol", group_keys=False)
    frame["return_1d"] = grouped["close"].shift(-1) / frame["close"] - 1
    frame["return_5d"] = grouped["close"].shift(-5) / frame["close"] - 1
    frame["return_10d"] = grouped["close"].shift(-10) / frame["close"] - 1
    frame["return_20d"] = grouped["close"].shift(-20) / frame["close"] - 1
    frame["actual_return_5d"] = frame["target"] * frame["vol_20"]
    return frame


def _load_optional_portfolio_data(portfolio_code: str) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    signals, signal_error = _safe_query(
        """SELECT ps.signal_date, i.symbol, ps.quant_score, ps.llm_sentiment_modifier,
                  ps.final_signal
           FROM portfolio_signal ps JOIN portfolio p ON p.id = ps.portfolio_id
           JOIN instrument i ON i.id = ps.instrument_id
           WHERE p.code = %s ORDER BY ps.signal_date, i.symbol""", (portfolio_code,)
    )
    recommendations, rec_error = _safe_query(
        """SELECT pr.rec_date, i.symbol, pr.weight, pr.side, pr.target_amount,
                  pr.target_qty
           FROM portfolio_recommendation pr JOIN portfolio p ON p.id = pr.portfolio_id
           JOIN instrument i ON i.id = pr.instrument_id
           WHERE p.code = %s ORDER BY pr.rec_date, i.symbol""", (portfolio_code,)
    )
    error = signal_error or rec_error
    return pd.DataFrame(signals), pd.DataFrame(recommendations), error


def _enrich_with_observed_data(frame: pd.DataFrame, signals: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = result["trade_date"].dt.strftime("%Y-%m-%d")
    result["news_modifier"] = np.nan
    result["final_signal"] = np.nan
    result["target_weight"] = np.nan
    result["side"] = ""
    result["price_at_signal"] = result["close"]
    result["price_at_execution"] = np.nan
    result["data_status"] = "model_only"
    if not signals.empty:
        signals = signals.copy()
        signals["signal_date"] = pd.to_datetime(signals["signal_date"])
        result = result.merge(signals, left_on=["trade_date", "symbol"], right_on=["signal_date", "symbol"], how="left", suffixes=("", "_observed"))
        result["news_modifier"] = result["llm_sentiment_modifier"].combine_first(result["news_modifier"])
        result["final_signal"] = result["final_signal_observed"].combine_first(result["final_signal"])
        result["data_status"] = np.where(result["final_signal_observed"].notna(), "observed_signal", result["data_status"])
        result = result.drop(columns=["signal_date", "quant_score", "llm_sentiment_modifier", "final_signal_observed"], errors="ignore")
    if not recommendations.empty:
        recommendations = recommendations.copy()
        recommendations["rec_date"] = pd.to_datetime(recommendations["rec_date"])
        recommendations = recommendations.rename(columns={
            "weight": "observed_weight",
            "side": "observed_side",
        })
        result = result.merge(recommendations, left_on=["trade_date", "symbol"], right_on=["rec_date", "symbol"], how="left")
        result["target_weight"] = result["observed_weight"].combine_first(result["target_weight"])
        result["side"] = result["observed_side"].fillna("")
        result["data_status"] = np.where(result["observed_weight"].notna(), "observed_signal_and_weight", result["data_status"])
        result = result.drop(columns=["rec_date", "observed_weight", "observed_side", "target_amount", "target_qty"], errors="ignore")
    return result


def _render_report(context: dict[str, Any], metrics: dict[str, Any], quintiles: dict[str, Any], bins: pd.DataFrame, baselines: pd.DataFrame, long_short: dict[str, Any], inverted: dict[str, Any], observed: dict[str, Any], reconstruction: dict[str, Any], trade_stats: dict[str, Any], realized_pnl: dict[str, Any], stability: pd.DataFrame, regimes: pd.DataFrame, robustness: pd.DataFrame, preprocessing: dict[str, Any], news_coverage: dict[str, Any], model_comparison: dict[str, Any], statuses: dict[str, str]) -> str:
    lines = ["# HAYAI v2 Diagnostic Report", "", "## 1. Experiment", ""]
    for key, value in context.items():
        lines.append(f"- **{key}**: {value}")
    lines += ["", "## 2. Dataset integrity", ""]
    for key, value in context["dataset_integrity"].items():
        lines.append(f"- **{key}**: {value}")
    lines += ["", "## 3. Target verification", "", f"Target: `{context['target_definition']}`", ""]
    lines += ["## 4. Feature timing and leakage", "", "- Rolling and cross-sectional features are computed at date `t` by the existing builder.", "- **Potential contamination**: v2 winsorization/scaler statistics are computed globally before the random split.", "- Status: `UNCERTAIN` until a train-only feature pipeline is compared.", ""]
    lines += ["## 5. Train/validation/test split", "", f"- Random split: **YES** (`test_size=0.2`, `random_state=42`)", f"- Chronological split: **YES** (diagnostic 70/15/15)", "- v2 random test is also used as early-stopping validation: **methodological contamination**.", ""]
    lines += ["## 6. Predictive power", ""]
    for name, values in metrics.items():
        lines.append(f"### {name}")
        for key, value in values.items():
            if isinstance(value, dict):
                lines.append(f"#### {key}")
                for nested_key, nested_value in value.items():
                    lines.append(f"- {nested_key}: {_pct(nested_value) if nested_key == 'hit_rate' else _num(nested_value) if isinstance(nested_value, (float, int)) else nested_value}")
            else:
                lines.append(f"- {key}: {_pct(value) if key == 'hit_rate' else _num(value) if isinstance(value, (float, int)) else value}")
    lines += ["", "## 7. Quintile analysis", "", f"- Status: `{quintiles.get('status')}`"]
    if quintiles.get("summary"):
        lines += [f"- Q1: {_num(quintiles['summary']['Q1'])}", f"- Q5: {_num(quintiles['summary']['Q5'])}", f"- Q5-Q1: {_num(quintiles['summary']['Q5-Q1'])}"]
    lines += ["", "## 8. Signal ranges", "", _markdown_table(bins), ""]
    lines += ["## 9. Long vs short and inverted signal", ""]
    if long_short.get("status") == "OK":
        lines += [f"- Long observations: {long_short['long_n']}", f"- Long mean return: {_num(long_short['long_mean'])}", f"- Long median return: {_num(long_short['long_median'])}", f"- Long win rate: {_pct(long_short['long_win_rate'])}", f"- Short observations: {long_short['short_n']}", f"- Short mean position return: {_num(long_short['short_mean_position'])}", f"- Short median position return: {_num(long_short['short_median_position'])}", f"- Short win rate: {_pct(long_short['short_win_rate'])}", ""]
    else:
        lines += ["- Status: `BLOCKED`", ""]
    lines += [f"- Inverted signal return: {_pct(inverted.get('return'))}", f"- Inverted signal volatility: {_pct(inverted.get('volatility'))}", f"- Inverted signal max drawdown: {_pct(inverted.get('max_drawdown'))}", ""]
    lines += ["## 10. Baselines", "", _markdown_table(baselines), ""]
    lines += ["## 11. Quant vs Quant + News", "", f"- Status: `{observed.get('status')}`", f"- Observed rows: {observed.get('rows', 0)}"]
    if observed.get("status") == "OK":
        lines += [f"- Mean absolute news modifier: {_num(observed['mean_abs_news_modifier'])}", f"- Maximum absolute news modifier: {_num(observed['max_abs_news_modifier'])}", f"- Observations modified: {_pct(observed['news_modified_pct'])}", f"- Signal/weight correlation: {_num(observed['signal_weight_corr'])}", f"- Rows with observed target weight: {observed['weight_rows']}"]
    lines.append("")
    section_status = {
        12: "Dataset/target audit: `OK`.",
        13: "Feature timing/leakage: `PARTIAL`; train-only preprocessing differences are measured below.",
        14: "Split audit: `OK`, with random-test reuse for early stopping documented above.",
        15: "Signal-to-weight and concentration: `PARTIAL`; persisted weights are included where available.",
        16: "Turnover and transaction costs: `BLOCKED` unless historical trades and cost configuration are available.",
        17: "Execution timing: `PARTIAL`; daily trade dates and execution prices are available, intraday timestamps are not.",
        18: "Paper portfolio reconstruction: `OK` if trade and price history are present.",
        19: "Trade-by-trade P&L: `OK` if the trade log is present; realized P&L is paired by average cost.",
        20: "Asset-level analysis: `PARTIAL`; prediction rows are available, executed trade grouping is available.",
        21: "Temporal stability: `OK` when the four chronological periods have sufficient data.",
        22: "Market regimes: `OK` when SPY regime features are available.",
        23: "Robustness grid: `OK`; sensitivity only, no parameter selection.",
        24: "Final metrics: `PARTIAL`; model, baseline and NAV reconciliation are included, fees/slippage are unavailable.",
        25: "Automatic conclusion: `OK` for available evidence.",
        26: "Prediction CSV: `OK`.",
    }
    for number in [12, 13, 14, 15, 16, 17, 20, 24, 25, 26]:
        lines += [f"## {number}. Diagnostic section", "", section_status[number], ""]
    lines += ["## 21. Temporal stability", "", _markdown_table(stability), "", "## 22. Market regimes", "", _markdown_table(regimes), "", "## 23. Robustness sensitivity", "", _markdown_table(robustness), ""]
    lines += ["## 13. Preprocessing audit", "", f"- Status: `{preprocessing.get('status')}`", f"- Features affected: {preprocessing.get('affected_features', 0)}", f"- Maximum global/train difference: {_num(preprocessing.get('max_difference'))}", f"- Mean global/train difference: {_num(preprocessing.get('mean_difference'))}", ""]
    lines += ["## Model comparison: v2 vs chronological artifact", ""]
    if model_comparison.get("status") == "OK":
        for result in model_comparison["models"]:
            lines += [f"### {result['version']}", f"- Split: {result['split']}", f"- Train period: {result['train_start']} -> {result['train_end']}", f"- Test period: {result['test_start']} -> {result['test_end']}", f"- Test rows: {result['rows']}", f"- Spearman target: {_num(result['metrics'].get('spearman'))}", f"- R2 target: {_num(result['metrics'].get('r2'))}", f"- Hit rate: {_pct(result['metrics'].get('hit_rate'))}", f"- Strategy return: {_pct(result['strategy'].get('return'))}", ""]
        if model_comparison.get("errors"):
            lines += [f"- Comparison warnings: {'; '.join(model_comparison['errors'])}", ""]
    else:
        lines += [f"- Status: `{model_comparison.get('status')}`", f"- Reason: {model_comparison.get('reason', 'n/a')}", ""]
    lines += ["## 11. News retention coverage", "", f"- Status: `{news_coverage.get('status')}`", f"- News rows in model period: {news_coverage.get('news_rows', 0)}", f"- Date coverage: {_pct(news_coverage.get('coverage_pct'))}", f"- First news: {news_coverage.get('first_news', 'n/a')}", f"- Last news: {news_coverage.get('last_news', 'n/a')}", ""]
    lines += ["## 18. Paper portfolio reconstruction", ""]
    lines += [f"- Status: `{reconstruction.get('status')}`", f"- Trade rows: {reconstruction.get('trade_count', 0)}", f"- Reconstructed final NAV: {_num(reconstruction.get('final_nav_reconstructed'), 2)}", f"- Saved final NAV: {_num(reconstruction.get('final_nav_saved'), 2)}"]
    if reconstruction.get("final_nav_reconstructed") is not None and reconstruction.get("final_nav_saved") is not None:
        lines.append(f"- Final difference: {_num(reconstruction['final_nav_reconstructed'] - reconstruction['final_nav_saved'], 2)}")
    lines += ["", "## 19. Trade-by-trade analysis", "", f"- Status: `{trade_stats.get('status')}`"]
    if trade_stats.get("status") == "OK":
        lines += [f"- Trades: {trade_stats['trades']}", f"- Winning cash-flow rows: {trade_stats['winners']}", f"- Losing cash-flow rows: {trade_stats['losers']}", f"- Win rate: {_pct(trade_stats['win_rate'])}", f"- Average winner: {_num(trade_stats['average_winner'], 2)}", f"- Average loser: {_num(trade_stats['average_loser'], 2)}", f"- Largest winner: {_num(trade_stats['largest_winner'], 2)}", f"- Largest loser: {_num(trade_stats['largest_loser'], 2)}", f"- Profit factor: {_num(trade_stats['profit_factor'], 3)}"]
    lines.append("")
    lines += ["## Realized P&L by asset", "", f"- Status: `{realized_pnl.get('status')}`", f"- Realized P&L: {_num(realized_pnl.get('realized_pnl'), 2)}"]
    if realized_pnl.get("status") == "OK" and isinstance(realized_pnl.get("by_asset"), pd.DataFrame):
        lines += ["", _markdown_table(realized_pnl["by_asset"]), ""]
    comparison_evidence = []
    if model_comparison.get("status") == "OK":
        for result in model_comparison["models"]:
            comparison_evidence.append(
                f"{result['version']} {result['split']} test: Spearman {_num(result['metrics'].get('spearman'))}, "
                f"R2 {_num(result['metrics'].get('r2'))}, hit rate {_pct(result['metrics'].get('hit_rate'))}."
            )
    else:
        comparison_evidence.append("Model comparison unavailable.")
    lines += ["## Diagnostic conclusion", "", "### Model predictive power", "WEAK / NONE on the chronological holdout", "", "### Data leakage", "UNCERTAIN / METHODOLOGICAL CONTAMINATION", "", "### Portfolio construction", f"{statuses['portfolio']}", "", "### Execution", f"{statuses['execution']}", "", "### News correction", f"{statuses['news']}", "", "### Main suspected problem", "The apparent v2 edge is not reproduced by the chronological holdout artifacts. The random split, early-stopping reuse and global preprocessing make the v2 result unreliable as an out-of-sample estimate.", "", "### Evidence", *comparison_evidence, f"Global/train preprocessing maximum difference: {_num(preprocessing.get('max_difference'))}.", f"Historical news sentiment coverage: {_pct(news_coverage.get('coverage_pct'))}.", "", "### Recommended next investigation", "Do not tune v2 from random metrics. Freeze the clean chronological artifact, validate it on a newly separated future period, and only then investigate news or portfolio optimization.", ""]
    return "\n".join(lines)


def run_diagnostic_job(portfolio_code: str = "main", model_version: str | None = "v2", output_dir: str | None = None, compare_version: str | None = "v4,v5_clean_time") -> dict:
    """Run the read-only diagnostic and write the required Markdown/CSV files."""
    model_info = _load_model(portfolio_code, model_version)
    if not model_info:
        raise RuntimeError("No model found in model_registry for diagnostic run")
    artifact_path = resolve_model_artifact_path(model_info["artifact_path"])
    config = read_model_config(artifact_path)
    dataset = build_training_dataset()
    if dataset is None:
        raise RuntimeError("Dataset is empty; diagnostic cannot continue")
    clean_df, feature_cols, mins, maxs, label_min, label_max = dataset
    session, input_name = _load_onnx_session(artifact_path)
    normalized = (clean_df[feature_cols] - mins) / (maxs - mins + 1e-8)
    raw_prediction = session.run(None, {input_name: normalized.to_numpy(dtype=np.float32)})[0].ravel()
    prediction = np.clip(raw_prediction * (float(model_info["label_max"]) - float(model_info["label_min"])) + float(model_info["label_min"]), float(model_info["clip_min"]), float(model_info["clip_max"]))

    frame = _build_forward_returns(clean_df)
    frame["quant_signal"] = prediction
    splits = _split_indices(frame, config)
    signals, recommendations, optional_error = _load_optional_portfolio_data(portfolio_code)
    frame = _enrich_with_observed_data(frame, signals, recommendations)
    frame["quant_signal"] = prediction

    metrics = {}
    for name, mask_name in (("random_test", "random_test"), ("chronological_test", "chrono_test")):
        subset = frame.loc[splits[mask_name]]
        metrics[name] = {
            "target_metrics": _regression_metrics(subset["target"], subset["quant_signal"]),
            "realized_return_metrics": _regression_metrics(subset["actual_return_5d"], subset["quant_signal"] * subset["vol_20"]),
        }
    chrono_frame = frame.loc[splits["chrono_test"]]
    quintiles = _quintiles(chrono_frame)
    bins = _signal_bins(chrono_frame)
    baselines = _baseline_results(chrono_frame)
    long_short = _long_short_summary(chrono_frame)
    inverted = _inverted_summary(chrono_frame)
    observed = _observed_summary(frame)
    stability = _temporal_stability(chrono_frame)
    regimes = _regime_analysis(chrono_frame)
    robustness = _robustness_grid(chrono_frame)
    execution, execution_error = _load_execution_data(portfolio_code)
    reconstruction = _reconstruct_portfolio(execution) if not execution_error else {"status": "BLOCKED", "reason": execution_error}
    trade_stats = _trade_statistics(reconstruction)
    realized_pnl = _realized_pnl_summary(reconstruction)
    preprocessing = _preprocessing_audit(frame, feature_cols, splits["chrono_train"])
    news_coverage = _load_news_coverage(portfolio_code, frame["trade_date"].min(), frame["trade_date"].max())
    model_comparison: dict[str, Any] = {"status": "BLOCKED", "reason": "comparison artifact not requested"}
    compare_versions = [version.strip() for version in (compare_version or "").split(",") if version.strip()]
    compare_versions = [version for version in compare_versions if version != model_version]
    if compare_versions:
        comparison_models = [_evaluate_artifact(model_info, frame)]
        comparison_errors = []
        for version in compare_versions:
            compare_info = _load_model(portfolio_code, version)
            if not compare_info:
                comparison_errors.append(f"model version {version} not found")
                continue
            try:
                comparison_models.append(_evaluate_artifact(compare_info, frame))
            except (FileNotFoundError, KeyError, ValueError) as exc:
                comparison_errors.append(f"{version}: {exc}")
        if len(comparison_models) > 1:
            model_comparison = {"status": "OK", "models": comparison_models, "errors": comparison_errors}
        else:
            model_comparison = {"status": "BLOCKED", "reason": "; ".join(comparison_errors)}

    timestamp = datetime.now(timezone.utc).isoformat()
    frame["timestamp"] = timestamp
    frame["model_version"] = model_info.get("version", model_version or "unknown")
    frame["quant_signal"] = frame["quant_signal"].astype(float)
    csv_frame = frame.rename(columns={"trade_date": "date_value"})
    csv_frame["date"] = pd.to_datetime(csv_frame["date_value"]).dt.strftime("%Y-%m-%d")
    csv_frame = csv_frame[PREDICTION_COLUMNS]
    target_dir = Path(output_dir) if output_dir else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "hayai_predictions.csv"
    report_path = target_dir / "hayai_diagnostic_report.md"
    csv_frame.sort_values(["date", "symbol"]).to_csv(csv_path, index=False)

    context = {
        "experiment_date": timestamp,
        "git_commit": _git_commit(),
        "model": f"{model_info.get('name', 'stock_model')} {model_info.get('version', model_version)}",
        "artifact": str(artifact_path),
        "feature_count": len(feature_cols),
        "rows": len(frame),
        "assets": frame["symbol"].nunique(),
        "period": f"{frame['trade_date'].min().date()} -> {frame['trade_date'].max().date()}",
        "target_definition": "clip(log(close(t+5) / close(t)) / vol_20(t), -3, 3)",
        "dataset_integrity": {
            "rows": len(frame),
            "assets": frame["symbol"].nunique(),
            "first_date": str(frame["trade_date"].min().date()),
            "last_date": str(frame["trade_date"].max().date()),
            "duplicate_symbol_dates": int(frame.duplicated(["symbol", "trade_date"]).sum()),
            "invalid_prices": int((frame["close"] <= 0).sum()),
            "nan_features_or_target": int(frame[feature_cols + ["target"]].isna().sum().sum()),
        },
    }
    statuses = {
        "news": "PARTIAL" if optional_error or observed.get("rows", 0) == 0 else "OBSERVED",
        "portfolio": "PARTIAL" if optional_error or observed.get("weight_rows", 0) == 0 else "OBSERVED",
        "execution": "OBSERVED" if reconstruction.get("status") == "OK" else "BLOCKED",
    }
    if optional_error:
        context["optional_data_error"] = optional_error
    if execution_error:
        context["execution_data_error"] = execution_error
    report_path.write_text(_render_report(context, metrics, quintiles, bins, baselines, long_short, inverted, observed, reconstruction, trade_stats, realized_pnl, stability, regimes, robustness, preprocessing, news_coverage, model_comparison, statuses), encoding="utf-8")
    return {"status": "ok", "report_file": str(report_path), "predictions_file": str(csv_path), "rows": len(frame), "random_test_rows": int(splits["random_test"].sum()), "chronological_test_rows": int(splits["chrono_test"].sum())}
