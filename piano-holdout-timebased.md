# Piano Operativo: Holdout Time-Based per Validare il Modello (v4)

Obiettivo: validare l'architettura/feature set (quello di v2) con un **vero holdout
cronologico** che non venga mai visto dall'early stopping. Split 70/15/15 per date:
train = 70% piÃ¹ vecchio, val = 15% centrale (solo per early stopping), test = 15% piÃ¹
recente (mai usato per il training). Il modello v4 viene addestrato cosÃ¬ e valutato con
`verify` e `backtest` sul solo test cronologico, rimuovendo il bias dell'early stopping.

---

## Task 1: dataset_builder â€” helper di split cronologico
- **Stato**: done
- **Scopo**: funzioni riutilizzabili per splittare per data e riprodurre gli stessi cutoff.
- **Risultato atteso**: `split_by_date()`, `split_by_cutoffs()`, `read_model_config()`.
- **Todolist**:
  - [x] `split_by_date(clean_df, 0.15, 0.15)` â†’ mask train/val/test + cutoff date
  - [x] `split_by_cutoffs(clean_df, train_end, val_end)` per riprodurre lo split
  - [x] `read_model_config(artifact_path)` per leggere split e cutoff da config.json

## Task 2: training â€” split time-based opzionale
- **Stato**: done
- **Scopo**: addestrare con split cronologico senza rompere il flusso 'random' esistente.
- **Risultato atteso**: `build_dataset_and_train(split='time', version='v4', make_active=False)`
  con min/max e label fit solo sul train, early stopping su val, cutoff salvati in config.json,
  registrazione v4 come 'draft' (non attivo).
- **Todolist**:
  - [x] Parametrizzare `split`, `version`, `make_active` in `build_dataset_and_train`
  - [x] Logica split 'time': scaler sul train, validation_data=val, cutoff in config.json
  - [x] Registrare v4 come draft (v2 resta attivo)

## Task 3: verify/backtest â€” parametro modello e split cronologico
- **Stato**: done
- **Scopo**: valutare un modello specifico (es. v4) riproducendo lo split salvato in config.
- **Risultato atteso**: `run_verify_model_job(portfolio_code, model_version=None)` e
  `run_backtest_job(portfolio_code, top_n, bottom_n, model_version=None)` che, se il config
  del modello ha `split='time'`, usano `split_by_cutoffs` e valutano solo sul test cronologico
  (scaler fittato sul train).
- **Todolist**:
  - [x] Aggiungere `model_version` a verify e backtest (`_load_model`)
  - [x] Rilevare `split` da config.json e applicare `split_by_cutoffs`
  - [x] Fittare min/max e label solo sul train in modalitÃ  time

## Task 4: CLI â€” parametro `--version`
- **Stato**: done
- **Scopo**: permettere `python -m app.cli verify --version v4`.
- **Risultato atteso**: nuovo argomento `--version` passato ai job che lo accettano.
- **Todolist**:
  - [x] Aggiungere `--version` in `app/cli.py` e passarlo se in firma
  - [x] Smoke test dei comandi

## Task 5: eseguire l'holdout e confrontare
- **Stato**: done
- **Scopo**: produrre i numeri out-of-sample veri di v4 e confrontarli con v2 (ottimistico).
- **Risultato atteso**: v4 draft addestrato; `verify --version v4` e `backtest --version v4`;
  tabella confronto v2 vs v4; decisione su eventuale promozione a produzione.
- **Todolist**:
  - [x] Addestrare v4 (time split, draft)
  - [x] `verify --version v4` e `backtest --version v4`
  - [x] Confronto e commento (con decisione se promuovere)

## Task 6: Documentazione
- **Stato**: done
- **Scopo**: documentare la metodologia di validazione.
- **Risultato atteso**: `doc-new-app/03-ml-pipeline.md` con la sezione holdout time-based.
- **Todolist**:
  - [x] Aggiungere sezione holdout time-based in `03-ml-pipeline.md`


