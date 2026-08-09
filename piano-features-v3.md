# Piano Operativo: Feature Giorno-Settimana e Giorni dall'Ultimo Max (v3)

Aggiunta di due feature all'ultima parte del dataset (27 totali): giorno della settimana in
encoding ciclico (`dow_sin`/`dow_cos`) e giorni di trading dall'ultimo massimo del prezzo di
chiusura (`days_since_high`, finestra 252gg, log1p). Retraining come **v3**, confronto con v2
tramite `verify` e `backtest`.

---

## Task 1: dataset_builder â€” nuove feature
- **Stato**: done
- **Scopo**: aggiungere `dow_sin`, `dow_cos`, `days_since_high` al pannello.
- **Risultato atteso**: `FEATURE_COLS` a 27 elementi; pannello con le nuove colonne senza NaN;
  `compute_panel_features` resta l'unica fonte per training e predict.
- **Todolist**:
  - [x] In `_add_cross_sectional_features`: `dow_sin`/`dow_cos` da `trade_date.dt.dayofweek`
  - [x] In `compute_features`: `days_since_high` = `log1p(pos - last_high_pos)` con
    `rolling(252, min_periods=20).max()` e ffill dell'ultima posizione di max
  - [x] Aggiornare `FEATURE_COLS`

## Task 2: training v3
- **Stato**: done
- **Scopo**: riaddestrare con le nuove feature e registrare v3.
- **Risultato atteso**: `MODEL_VERSION="v3"`, artifact in `model/stock_model/v3`, v3 attivo nel
  portfolio, v2 mantenuto per rollback.
- **Todolist**:
  - [x] Impostare `MODEL_VERSION = "v3"`
  - [x] Eseguire `build_dataset_and_train()` (senza re-download)
  - [x] Verificare registrazione v3 attiva

## Task 3: misurazione e confronto
- **Stato**: done
- **Scopo**: valutare l'impatto delle nuove feature.
- **Risultato atteso**: report `verify` e `backtest` su v3; tabella confronto v1/v2/v3.
- **Todolist**:
  - [x] Eseguire `python -m app.cli verify`
  - [x] Eseguire `python -m app.cli backtest`
  - [x] Confrontare con v2 e commentare

## Task 4: Documentazione
- **Stato**: done
- **Scopo**: aggiornare il doc del modello.
- **Risultato atteso**: `doc-new-app/03-ml-pipeline.md` con le 27 feature e i risultati v3.
- **Todolist**:
  - [x] Aggiornare la lista feature in `03-ml-pipeline.md`
  - [x] Aggiungere i risultati v3


