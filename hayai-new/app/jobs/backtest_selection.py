import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.model_selection import train_test_split

from app.config import settings
from app.jobs.dataset_builder import build_training_dataset
from app.jobs.verify_model import _load_active_model, _load_onnx_session
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.backtest_selection")

DEFAULT_MODEL_DIR = Path("model/stock_model/v1")
REBALANCE_STEP = 5


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Cross-sectional rank correlation without scipy."""
    mask = a.notna() & b.notna()
    a = a[mask]
    b = b[mask]
    if len(a) < 3:
        return float("nan")
    ra = a.rank().to_numpy(dtype=float)
    rb = b.rank().to_numpy(dtype=float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _per_date_stats(sub: pd.DataFrame, top_n: int, bottom_n: int) -> dict:
    if len(sub) < top_n + bottom_n + 2:
        return None
    top = sub.nlargest(top_n, "quant_score")
    bottom = sub.nsmallest(bottom_n, "quant_score")
    spy = sub[sub["symbol"] == "SPY"]["fwd_log_ret"]
    return {
        "date": sub["trade_date"].iloc[0],
        "n": int(len(sub)),
        "long": float(top["fwd_log_ret"].mean()),
        "short": float(bottom["fwd_log_ret"].mean()),
        "universe": float(sub["fwd_log_ret"].mean()),
        "spy": float(spy.iloc[0]) if len(spy) else float("nan"),
        "spread": float(top["fwd_log_ret"].mean() - bottom["fwd_log_ret"].mean()),
        "spearman": _spearman(sub["quant_score"], sub["fwd_log_ret"]),
        "long_positive": bool(top["fwd_log_ret"].mean() > 0),
        "short_negative": bool(bottom["fwd_log_ret"].mean() < 0),
        "long_beats_universe": bool(top["fwd_log_ret"].mean() > sub["fwd_log_ret"].mean()),
    }


def run_backtest_job(portfolio_code: str = "main", top_n: int = 5, bottom_n: int = 5) -> dict:
    logger.info(f"Running selection backtest for portfolio '{portfolio_code}' (top={top_n}, bottom={bottom_n})...")

    model_info = _load_active_model(portfolio_code)
    if not model_info:
        logger.error("No active model found in model_registry. Skipping backtest.")
        return {"status": "no_active_model", "report_file": None}

    model_name = model_info.get("name", "stock_model")
    model_version = model_info.get("version", "v1")
    artifact_path = Path(model_info["artifact_path"])
    feature_cols = json.loads(model_info["feature_columns"])
    label_min_model = float(model_info["label_min"])
    label_max_model = float(model_info["label_max"])
    clip_min = float(model_info["clip_min"])
    clip_max = float(model_info["clip_max"])

    if not artifact_path.exists():
        logger.warning(f"Artifact path {artifact_path} not found, falling back to {DEFAULT_MODEL_DIR}")
        artifact_path = DEFAULT_MODEL_DIR

    ort_session, input_name = _load_onnx_session(artifact_path)

    logger.info("Building dataset from database...")
    dataset = build_training_dataset()
    if dataset is None:
        logger.error("Dataset build failed. Skipping backtest.")
        return {"status": "dataset_empty", "report_file": None}

    clean_df, _, mins, maxs, label_min_rec, label_max_rec = dataset

    X = clean_df[feature_cols]
    y = clean_df["target"]
    X_norm = (X - mins) / (maxs - mins + 1e-8)
    y_norm = (y - label_min_rec) / (label_max_rec - label_min_rec + 1e-8)

    _, X_test, _, _ = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
    test_mask = clean_df.index.isin(X_test.index)

    logger.info("Running ONNX inference over the whole panel...")
    raw_pred = ort_session.run(None, {input_name: X_norm.to_numpy(dtype=np.float32)})[0].ravel()
    pred_denorm = raw_pred * (label_max_model - label_min_model) + label_min_model
    quant_score = np.clip(pred_denorm, clip_min, clip_max)

    panel = clean_df.copy()
    panel["quant_score"] = quant_score
    panel["fwd_log_ret"] = panel["target"] * panel["vol_20"]
    panel["is_test"] = test_mask
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])

    def _aggregate(p: pd.DataFrame) -> dict:
        stats = []
        for _, sub in p.sort_values("trade_date").groupby("trade_date"):
            s = _per_date_stats(sub, top_n, bottom_n)
            if s:
                stats.append(s)
        df = pd.DataFrame(stats)
        if df.empty:
            return None
        return {
            "dates": int(len(df)),
            "long": float(df["long"].mean()),
            "short": float(df["short"].mean()),
            "universe": float(df["universe"].mean()),
            "spy": float(df["spy"].mean(skipna=True)),
            "spread": float(df["spread"].mean()),
            "spearman": float(df["spearman"].mean(skipna=True)),
            "hit_long_positive": float(df["long_positive"].mean()),
            "hit_short_negative": float(df["short_negative"].mean()),
            "hit_long_beats_universe": float(df["long_beats_universe"].mean()),
            "cumulative_spread": float(df["spread"].sum()),
        }

    test_stats = _aggregate(panel[panel["is_test"]])
    all_stats = _aggregate(panel)

    # Non-overlapping rebalancing (every REBALANCE_STEP trading days) on the test panel
    test_panel = panel[panel["is_test"]]
    dates = sorted(test_panel["trade_date"].unique())
    rebal_dates = dates[::REBALANCE_STEP]
    non_overlap = _aggregate(test_panel[test_panel["trade_date"].isin(rebal_dates)])

    report_lines = _build_report(
        model_name=model_name,
        model_version=model_version,
        artifact_path=str(artifact_path),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        portfolio=portfolio_code,
        top_n=top_n,
        bottom_n=bottom_n,
        test_stats=test_stats,
        all_stats=all_stats,
        non_overlap=non_overlap,
        test_min=pd.to_datetime(test_panel["trade_date"]).min().strftime("%Y-%m-%d"),
        test_max=pd.to_datetime(test_panel["trade_date"]).max().strftime("%Y-%m-%d"),
        panel_rows=int(len(test_panel)),
        rebalances=len(rebal_dates),
    )

    report_dir = settings.LOGS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"model_backtest_{model_name}_{model_version.lstrip('v')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_lines)

    for line in report_lines.splitlines():
        if line.strip():
            logger.info(line)

    logger.info(f"Backtest completed. Report written to {report_file}")
    return {
        "status": "ok",
        "report_file": str(report_file),
        "test_dates": test_stats["dates"] if test_stats else 0,
        "long_mean": test_stats["long"] if test_stats else None,
        "short_mean": test_stats["short"] if test_stats else None,
        "spread_mean": test_stats["spread"] if test_stats else None,
        "spearman_mean": test_stats["spearman"] if test_stats else None,
    }


def _fmt_pct(x) -> str:
    return f"{x * 100:+.2f}%" if x is not None and np.isfinite(x) else "  n/a"


def _fmt_num(x) -> str:
    return f"{x:+.4f}" if x is not None and np.isfinite(x) else "  n/a"


def _stats_block(title: str, s: dict) -> str:
    if not s:
        return f"{title}: nessun dato"
    lines = [
        title,
        f"  Date di ribilanciamento : {s['dates']}",
        f"  Ritorno medio LONG (top-N)        : {_fmt_num(s['long'])}  (log-ret 5gg)",
        f"  Ritorno medio SHORT (bottom-N)    : {_fmt_num(s['short'])}  (log-ret 5gg)",
        f"  Ritorno medio UNIVERSE (pari-peso): {_fmt_num(s['universe'])}  (log-ret 5gg)",
        f"  Ritorno medio SPY                 : {_fmt_num(s['spy'])}  (log-ret 5gg)",
        f"  Spread LONG - SHORT               : {_fmt_num(s['spread'])}",
        f"  Spearman medio (quant_score vs ret): {_fmt_num(s['spearman'])}",
        f"  Hit-rate LONG > 0                 : {_fmt_pct(s['hit_long_positive'])}",
        f"  Hit-rate SHORT < 0                : {_fmt_pct(s['hit_short_negative'])}",
        f"  Hit-rate LONG > UNIVERSE          : {_fmt_pct(s['hit_long_beats_universe'])}",
        f"  Spread cumulato                   : {_fmt_num(s['cumulative_spread'])}",
    ]
    return "\n".join(lines)


def _build_report(model_name, model_version, artifact_path, timestamp, portfolio,
                  top_n, bottom_n, test_stats, all_stats, non_overlap,
                  test_min, test_max, panel_rows, rebalances) -> str:
    sep = "=" * 78
    lines = [
        sep,
        "REPORT BACKTEST SELEZIONE LONG/SHORT — HAYAI v2",
        sep,
        f"Modello      : {model_name} v{model_version.lstrip('v')}",
        f"Artifact     : {artifact_path}",
        f"Portfolio    : {portfolio}",
        f"Data/ora     : {timestamp}",
        f"Strategia    : LONG top-{top_n} / SHORT bottom-{bottom_n} per quant_score",
        f"Target       : ritorno forward a 5 giorni (log-ret; quant_score in unita' ln-return/vol_20)",
        "",
        f"PANNELLO TEST (out-of-sample, 20% split random_state=42)",
        f"  Righe test : {panel_rows}",
        f"  Periodo    : {test_min} -> {test_max}",
        f"  Ribilanciamento non sovrapposto: ogni {REBALANCE_STEP} giorni di trading ({rebalances} date)",
        "",
        "NOTA METODOLOGICA:",
        "  - Il pannello test contiene solo righe con target noto (senza look-ahead).",
        "  - Le date sovrapposte (t->t+5 e t+1->t+6) sono correlate: le statistiche 'tutte le date'",
        "    sono indicative; il P&L cumulato usa solo ribilanciamenti non sovrapposti.",
        "  - SPY potrebbe mancare in alcune date (NaN esclusi dalla media).",
        "",
    ]

    lines += [sep, "1) RISULTATI OUT-OF-SAMPLE (solo test set)", sep]
    lines += _stats_block("   TUTTE LE DATE TEST (sovrapposte):", test_stats).splitlines()
    lines.append("")
    lines += _stats_block("   DATE NON SOVRAPPOSTE (ogni 5, P&L cumulato):", non_overlap).splitlines()
    lines.append("")
    lines += [sep, "2) RIFERIMENTO IN-SAMPLE (tutti i dati, ottimista)", sep]
    lines += _stats_block("   TUTTE LE DATE (train+test):", all_stats).splitlines()
    lines.append("")

    lines += [sep, "3) LETTURA RISULTATI", sep]
    lines += [
        "  - Spread LONG-SHORT medio > 0  -> il ranking del quant_score separa vincenti e perdenti.",
        "  - Spearman medio ~ 0            -> nessun edge cross-sezionale (selezione ~casuale).",
        "  - Hit-rate LONG > UNIVERSE      -> il top-N batte l'equal-weight dell'universo.",
        "  - Confronta LONG/SHORT con UNIVERSE e SPY: la selezione deve batterli per avere valore.",
        "  - Attenzione: le stime in-sample (sezione 2) sono ottimistiche perche' il modello ha",
        "    gia' visto quei dati durante il training.",
        "",
        sep,
        "FINE REPORT",
        sep,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    portfolio = sys.argv[1] if len(sys.argv) > 1 else "main"
    run_backtest_job(portfolio_code=portfolio)
