# Piano Operativo: Verifica del Modello (null/NaN, split, confronto, metriche)

Questo documento definisce il piano per aggiungere una verifica standalone del modello
ML deployato: controllo assenza di null/NaN nel dataset di training, verifica dello
split 80/20, confronto predizione vs actual su 100 righe casuali del test set e
produzione di un report con metriche di regressione (RMSE, MAE, R2, directional hit-rate)
e relative definizioni.

---

## Task 1: Modulo dataset condiviso
- **Stato**: done
- **Scopo**: estrarre la costruzione del dataset (query DB + feature engineering + dropna
  + min/max) da `build_dataset_and_train()` in un modulo riutilizzabile, garantendo che
  training e verifica usino feature identiche.
- **Risultato atteso**: nuovo `app/jobs/dataset_builder.py` con `build_training_dataset()`
  che ritorna `clean_df` (con `symbol` e `trade_date`), `FEATURE_COLS`, `mins`, `maxs`,
  `label_min`, `label_max`. `train_universe_pipeline.py` rifattorizzato per usarlo senza
  cambiare comportamento.
- **Test**: `python -m app.jobs.verify_model` (deve usare lo stesso dataset) e smoke test
  di import.
- **Todolist**:
  - [x] Creare `app/jobs/dataset_builder.py` con `FEATURE_COLS` e `build_training_dataset()`
  - [x] Includere nel builder il log delle righe scartate dal `dropna`
  - [x] Rifattorizzare `build_dataset_and_train()` per chiamare il builder
  - [x] Verificare che il modello venga ricostruito identico (smoke test)

## Task 2: Script standalone `verify_model.py`
- **Stato**: done
- **Scopo**: eseguire le verifiche richieste sul modello deployato e produrre un report.
- **Risultato atteso**: `app/jobs/verify_model.py` con `run_verify_model_job(portfolio_code="main")`
  che produce un report in `logs/model_verification_v1_<data>.txt` + console + `hayai.log`.
- **Todolist**:
  - [x] Caricare il modello attivo da `model_registry` (fallback `model/stock_model/v1`)
  - [x] Verifica NaN: log righe raw/dopo dropna; assert 0 NaN in train/test e nel campione
  - [x] Split 80/20 con `random_state=42`, log dimensioni e percentuali
  - [x] Predizione ONNX su tutto il test set, denormalizzazione e clip
  - [x] Metriche RMSE, MAE, R2, directional hit-rate + baseline ingenua, con definizioni
  - [x] Campionare 100 righe casuali e produrre tabella `symbol | date | actual | prediction | match`
  - [x] Scrivere il report in italiano e loggare in console + `hayai.log`
  - [x] Warning di drift se min/max ricomputati differiscono dagli artifact

## Task 3: Registrazione CLI
- **Stato**: done
- **Scopo**: rendere la verifica richiamabile da CLI.
- **Risultato atteso**: job `verify` in `JOBS_MAP` (`app/cli.py`) + esecuzione diretta
  `python -m app.jobs.verify_model`.
- **Todolist**:
  - [x] Aggiungere `verify` a `JOBS_MAP` in `app/cli.py`
  - [x] Aggiungere `if __name__ == "__main__"` a `verify_model.py`

## Task 4: Documentazione
- **Stato**: done
- **Scopo**: allineare la documentazione alla nuova verifica.
- **Risultato atteso**: sezione "Verifica del modello" in `doc-new-app/03-ml-pipeline.md`
  e nota in `doc-new-app/07-operativita-batch.md`.
- **Todolist**:
  - [x] Aggiungere sezione verifica in `doc-new-app/03-ml-pipeline.md`
  - [x] Aggiungere nota sulla verifica manuale in `doc-new-app/07-operativita-batch.md`


