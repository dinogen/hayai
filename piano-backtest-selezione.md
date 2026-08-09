# Piano Operativo: Backtest della Selezione Long/Short (quant_score)

Questo documento definisce il piano per valutare se il ranking prodotto dal `quant_score`
(top long / bottom short) ha un edge reale sul mercato, usando il test set del modello
(out-of-sample). Obiettivo: capire se la parte quant dell'ibrido ha valore o se Ã¨ inerte.

---

## Task 1: Script backtest `backtest_selection.py`
- **Stato**: done
- **Scopo**: costruire un pannello (simbolo, data) sul test set, predire `quant_score` con il
  modello ONNX, e per ogni data di ribilanciamento confrontare il ritorno forward a 5 giorni di
  top-N (long), bottom-N (short), universo pari-peso e SPY.
- **Risultato atteso**: report `logs/model_backtest_stock_model_v1_<data>.txt` con medie dei
  ritorni, hit-rate, spread long-short, correlazione cross-sezionale (spearman) e P&L cumulato
  con ribilanciamento non sovrapposto.
- **Todolist**:
  - [x] Riusare `_load_active_model`/`_load_onnx_session` da `verify_model.py`
  - [x] Ricostruire dataset, riprodurre lo split 80/20 (random_state=42) e isolare il test set
  - [x] Predire `quant_score` su tutto il pannello e calcolare `fwd_log_ret = target * vol_20`
  - [x] Per data: top-N/bottom-N, ritorni long/short/universe/SPY, spearman pred-vs-actual
  - [x] Aggregare su tutte le date (overlap) e su date non sovrapposte (ogni 5)
  - [x] Scrivere il report in italiano e loggare in console + `hayai.log`

## Task 2: Registrazione CLI e documentazione
- **Stato**: done
- **Scopo**: rendere il backtest richiamabile e documentarlo.
- **Risultato atteso**: job `backtest` in `app/cli.py` + sezione in `doc-new-app/03-ml-pipeline.md`.
- **Todolist**:
  - [x] Aggiungere `backtest` a `JOBS_MAP`
  - [x] Aggiungere sezione in `03-ml-pipeline.md`
  - [x] Eseguire il backtest e commentare i risultati


