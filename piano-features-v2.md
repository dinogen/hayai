# Piano Operativo: Feature Cross-Sezionali, Regime, Winsorize e Fix Training (v2)

Obiettivo: migliorare l'edge del modello (oggi Spearman ~0.03) aggiungendo feature relative
(cross-sezionali) e di regime di mercato, winsorizzare gli outlier, correggere l'underfitting
(early stopping + piÃ¹ epochs), riaddestrare come **v2** e confrontare v1 vs v2 con `verify` e
`backtest`. Le nuove feature sono calcolate per l'intero pannello `(simbolo, data)`, quindi va
aggiornata anche l'inferenza (`predict`), che oggi lavora strumento-per-strumento.

---

## Task 1: dataset_builder â€” feature cross-sezionali, regime, winsorize
- **Stato**: done
- **Scopo**: arricchire le 12 feature base con feature relative e di regime, winsorizzare gli
  outlier, e rendere il calcolo riutilizzabile per l'inferenza.
- **Risultato atteso**: `compute_panel_features(raw_df)` che produce il pannello con le nuove
  feature (`x_rank_*`, `rel_mom*_spy`, `excess_ret_5`, `beta_20`, `mkt_ret_5/20`,
  `breadth_20`, `dispersion_20`); `build_training_dataset()` applica il winsorize
  (quantile 0.5/99.5) prima del min-max.
- **Todolist**:
  - [x] Aggiungere `ret_1` (rendimento 1g) in `compute_features`
  - [x] Creare `_add_cross_sectional_features()` (rank per data, RS vs SPY, excess, beta, regime)
  - [x] Creare `compute_panel_features()` (base + cross-sezionali + regime)
  - [x] Applicare winsorize in `build_training_dataset()` prima di min/max
  - [x] Aggiornare `FEATURE_COLS` con le nuove feature

## Task 2: training â€” versione v2, early stopping, piÃ¹ epochs
- **Stato**: done
- **Scopo**: correggere l'underfitting e registrare un modello v2 separato.
- **Risultato atteso**: `build_dataset_and_train()` usa `MODEL_VERSION="v2"`, salva in
  `model/stock_model/v2`, registra `(stock_model, v2)` in `model_registry`, imposta v2 attivo;
  training con `EarlyStopping` e piÃ¹ epochs.
- **Todolist**:
  - [x] Parametrizzare la versione e la dir degli artifact
  - [x] Aggiungere `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` e `epochs=50`
  - [x] Registrare v2 in `model_registry` e aggiornare `portfolio.model_id`

## Task 3: predict â€” inferenza su pannello cross-sezionale
- **Stato**: done
- **Scopo**: allineare l'inferenza giornaliera alle nuove feature (calcolate su tutto il
  pannello del portafoglio, non per singolo strumento).
- **Risultato atteso**: `run_predict_job` carica l'ultima riga per strumento da
  `compute_panel_features` (pannello completo) e predice in batch.
- **Todolist**:
  - [x] Sostituire il loop per-strumento con pannello unico + `compute_panel_features`
  - [x] Predire in batch e upsert su `model_prediction`

## Task 4: addestrare v2 e confrontare v1 vs v2
- **Stato**: done
- **Scopo**: misurare l'impatto con gli strumenti esistenti.
- **Risultato atteso**: v2 addestrato e attivo; report `verify` e `backtest` su v2; confronto
  con i numeri di v1 (RMSE, R2, hit-rate, Spearman, spread long-short).
- **Todolist**:
  - [x] Eseguire `build_dataset_and_train()` (senza re-download)
  - [x] Eseguire `verify` e `backtest` su v2
  - [x] Confrontare v1 vs v2 e commentare

## Task 5: Documentazione
- **Stato**: done
- **Scopo**: aggiornare la documentazione.
- **Risultato atteso**: `doc-new-app/03-ml-pipeline.md` aggiornato con le nuove feature, la v2
  e i risultati del confronto.
- **Todolist**:
  - [x] Aggiornare feature set e training in `03-ml-pipeline.md`
  - [x] Nota su inferenza cross-sezionale in `03-ml-pipeline.md`


