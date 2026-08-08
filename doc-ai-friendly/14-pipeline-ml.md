# 14 — Pipeline ML (training su PC, inferenza su Raspberry)

Questo documento descrive la **pipeline del modello** della nuova applicazione:
dall'estrazione dei dati, al **training su PC in Jupyter**, all'**artefatto**
deployato sul Raspberry, fino all'**inferenza batch** e al calcolo delle
**raccomandazioni** long/short.

Le formule di feature e di pesi sono riportate per riferimento dal legacy
(documenti `03` e `05`), con gli adattamenti necessari al multi-asset.

## 1. Principio guida

> Training e inferenza devono usare **le stesse feature, lo stesso ordine di
> colonne e gli stessi parametri di normalizzazione**. L'artefatto del modello
> contiene tutto ciò che serve a riprodurre la predizione, senza leggere il
> codice legacy.

## 2. Dataset di training (su PC)

### 2.1 Estrazione dati

- Dal database del Raspberry (o da export CSV/parquet) si estraggono:
  - prezzi OHLCV per l'**universo** di training (`instrument` tipo stock/etf/fx/
    bond_yield) da `price_daily`;
  - serie **forex** da `fx_rate`;
  - serie **indici** da `index_value`.
- Periodo di default: **5 anni**, intervallo giornaliero (coerente con legacy).

### 2.2 Feature set type-agnostic

Le feature legacy dipendono da `high/low` e da dummies country/sector (doc `03`).
La nuova pipeline usa un **feature set comune** a tutte le classi di asset,
calcolato per strumento:

| Feature | Formula (per strumento, per data) |
|---|---|
| `log_return` | `ln(close_t / close_{t-trd})` |
| `mom_5`, `mom_10`, `mom_20` | `pct_change(5/10/20)` di `close` |
| `vol_10`, `vol_20` | `std` di `log_return` su 10/20 |
| `vol_ratio` | `vol_10 / vol_20` |
| `zscore_20` | `(close - MA20) / STD20(close)` |
| `trend_50` | `(close - MA50) / MA50` |
| `vol_regime` | `vol_10 / vol_60` |
| `mom_vol_adj` | `mom_20 / vol_20` |
| `volume_shock` | `volume / MA20(volume)` (dove volume disponibile) |
| `hl_range`, `close_range` | solo se `high/low` disponibili (stocks/etf) |
| `fx_*`, `index_*` | chiuse di forex/indici (merge su data, precedenti alla data) |
| `target` | `clip(ln(close_{t+trd}/close_t) / vol_20, clip_min, clip_max)` |

Vincoli:
- **Niente look-ahead**: forex/indici e target usano solo dati disponibili alla
  data `t` (il target guarda avanti solo nella costruzione del dataset di training,
  non durante l'inferenza).
- **Niente dummies country/sector**: rendono il modello **type-agnostic**
  (decisione chiave, doc `11 §4.5`). Opzionalmente si può aggiungere una colonna
  `instrument_type` codificata se si vuole un modello unico sensibile alla classe.
- **Allineamento colonne**: l'ordine delle feature è fissato da
  `model_registry.feature_columns` (JSON). Le feature inesistenti per un tipo
  (es. `hl_range` per fx) vengono sostituite con 0 nella normalizzazione o escluse
  a seconda della strategia scelta (vedi §4.1).

### 2.3 Decisione modelli (aperta ma con raccomandazione)

Raccomandazione: **due opzioni supportate** dal design:
- **(A) modello unico type-agnostic** senza dummies (semplice, un solo artefatto);
- **(B) modello per classe di asset** (stocks/etf, fx, bond_yield) con artefatti
  separati e `portfolio.model_id` che sceglie quale usare.

Il database (doc `13`) supporta entrambe: `model_registry` permette più modelli e
`portfolio.model_id` ne seleziona uno. La scelta finale è un parametro di
configurazione, non un vincolo architetturale.

## 3. Training (Jupyter, su PC)

Flusso nel notebook `train_model_new.ipynb`:

1. **Load**: legge il dataset esportato (parquet/CSV).
2. **Pulizia**: `replace(±inf, NaN)`, `dropna`, rimozione di `date`, `symbol`.
3. **EDA** (facoltativo): distribuzione del target, correlazioni.
4. **Normalizzazione min-max** per colonna:
   - `maxs = df.max()`, `mins = df.min()`;
   - `df = (df - mins) / (maxs - mins)`;
   - `label_min = mins['target']`, `label_max = maxs['target']`.
5. **Split**: `train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)`.
6. **Architettura** (identica a legacy, doc `04`):
   - `Input(n_row)` → `Dense(100, relu)` → `Dense(80, relu)` → `Dense(20, relu)` →
     `Dense(1, sigmoid)`.
   - Optimizer `Adam`, loss `mean_squared_error`, `epochs`/`batch_size`/
     `validation_split` da configurazione.
7. **Valutazione**: RMSE, precision/recall/accuracy sul segno, correlazione.
8. **Salvataggio artefatto** (vedi §3.1).
9. **Export ONNX** (vedi §3.2).

### 3.1 Artefatto del modello

Struttura cartella `model/<name>/<version>/`:

```
model.onnx         # per l'inferenza sul Pi (onnxruntime)
model.keras        # per ri-uso/ri-training (formato Keras)
mins.csv           # normalizzazione per colonna
maxs.csv
config.json        # label_min, label_max, clip_min/max, feature_columns (ordine)
metadata.json      # data training, metriche, fingerprint dataset, backend
```

`config.json` e `metadata.json` sono il **contratto** tra training e inferenza:
l'inferenza legge le feature in `feature_columns` e i range da `mins/maxs`.

### 3.2 Export ONNX

- Nel notebook, dopo il training:
  `tf2onnx` o `keras` (via `keras.export()` con `ExportArchive`) produce
  `model.onnx` (input float32, shape `(None, n_row)`).
- La normalizzazione **non** è nel grafo: resta in `mins/maxs` (semplice e
  trasparente, come nel legacy).
- Alternativa se l'export ONNX desse problemi: inferenza con **Keras 3 + backend
  numpy** (il Pi non necessita di TensorFlow).

### 3.3 Registrazione

- `model_registry` viene aggiornata con `name`, `version`, `artifact_path`,
  `feature_columns`, `label_min/max`, `clip_min/max`, `metrics`,
  `dataset_fingerprint` (hash SHA-256 delle righe/colonne del dataset).
- `status='active'` → usato dai job `predict`/`recommend`.

## 4. Inferenza batch sul Raspberry

### 4.1 `job predict`

1. Carica `model.onnx` (onnxruntime) + `mins/maxs` + `config.json` del modello
   attivo (dal `model_registry` → `portfolio.model_id`).
2. Estrae l'**ultima data** con dati completi per ciascun strumento del portafoglio.
3. Calcola le feature (stesse funzioni del training) e normalizza con `mins/maxs`.
4. Esegue `sess.run` sull'ONNX.
5. Denormalizza: `pred = pred * (label_max - label_min) + label_min`.
6. Clip: `pred.clip(clip_min, clip_max)`.
7. **Upsert** in `prediction` (chiave: model+instrument+as_of_date).

### 4.2 Robustezza

- Feature mancanti (es. `hl_range` per fx): se la strategia è "inclusa con 0",
  si riempiono con 0 **dopo** la normalizzazione (o si normalizza con min/max del
  training). La coerenza è garantita dal `feature_columns` salvato.
- Strumento senza dati sufficienti (es. meno di 60 barre): **escluso** dalla
  predizione e tracciato nel `job_run`.
- Errore ONNX su uno strumento → log + continuazione.

## 5. Calcolo raccomandazioni (`job recommend`)

Riproduce la logica legacy (doc `05 §1`) adattata:

1. Input: `prediction` (ultima data) + `vol_20` + parametri del portafoglio.
2. `weight_raw = prediction.clip(clip_min, clip_max) / vol_20`.
3. Ordina per `weight_raw` decrescente.
4. Seleziona top `n_long` con peso > 0 (long) e bottom `n_short` con peso < 0
   (short).
5. `weight = weight_raw / sum(|weight_raw|)` con asserzione
   `0.99 < sum(|weight|) < 1.01`.
6. **Indicazione di posizione** (senza broker):
   - `equity_indicativa` (parametro di config, es. 100000);
   - `capital = equity_indicativa * risk_percentage`;
   - `target_amount = weight * capital`;
   - `target_qty = round(target_amount / price)` (prezzo corrente);
   - soglia `qty_diff_perc_min`: scarta ribilanciamenti trascurabili rispetto alla
     raccomandazione precedente.
7. **Upsert** in `recommendation` (chiave: portfolio+instrument+rec_date),
   salvando `prev_weight` per visualizzare la variazione.

### 5.1 Gestione multi-asset nei pesi

- Le predizioni di **classi diverse** (azioni, ETF, forex, bond) sono comparabili
  perché il target è normalizzato (`rendimento/vol`).
- Se si adotta l'opzione **(B)** (modelli per classe), i pesi vengono calcolati
  aggregando le predizioni delle diverse classi nel ranking unico del portafoglio;
  in alternativa si può allocare per classe (parametro di configurazione).

## 6. Riproducibilità

- L'artefatto include il **fingerprint del dataset** e l'**ordine delle feature**:
  predizioni riproducibili a parità di dati e modello.
- Il `job predict` scrive `features_hash` per tracciabilità.
- I parametri di raccomandazione sono in `portfolio` (non hardcoded).

## 7. Requisiti soddisfatti

- RF-30 → §2-3 (training su PC, jupyter).
- RF-31/RF-32 → §3.1-3.3 (artefatto completo + registry).
- RF-33/RF-34 → §4 (inferenza batch + denormalizzazione/clip).
- RF-40/41/42/43 → §5 (raccomandazioni long/short, solo indicazioni).
- RN-02 (riproducibilità) → §6.
