# 04 — Modello predittivo

Questo documento descrive il **training del modello** (dal notebook
`train_model.ipynb`) e la fase di **inference** (`hayai_bo.apply_prediction`).

Il modello è una **rete neurale densa (MLP)** addestrata con Keras/TensorFlow per
prevedere il **target normalizzato** (rendimento futuro normalizzato per volatilità).

## 1. Dataset di training

Fonte: `f001_features.parquet` del **portfolio modello** (id che inizia con `model_`).

Passi nel notebook:

1. `df = read_parquet(FILE_FEATURES)` (nel `portfolio_dir` del modello).
2. `df = df.replace([inf, -inf], NaN)`; `df = df.dropna()`.
3. **EDA**: `df.describe()`, `df.info()`, istogramma del target (classi da 10),
   heatmap di correlazione (seaborn).
4. Drop di `date`, `symbol` (rimangono solo le feature + `target`).
5. Controllo: se ci sono `NaN` → `ValueError`.

> Il notebook è il punto in cui vengono salvati i parametri di normalizzazione
> (`mins.csv`, `maxs.csv`) e la `label_min`/`label_max` nel `conf.ini` del portfolio
> modello (sezione `[predictions]`).

## 2. Normalizzazione

Per **ogni colonna** (feature + target):

```
maxs = df.max()
mins = df.min()
df = (df - mins) / (maxs - mins)
label_min = mins['target']
label_max = maxs['target']
```

- `maxs` e `mins` vengono salvati come `maxs.csv` / `mins.csv` (indice `col`,
  colonna `value`) nel portfolio **modello**.
- `label_min`, `label_max` salvati in `conf.ini` → `[predictions]`.
- Controllo: se in `maxs`/`mins` ci sono `±inf` → `ValueError`.

> In fase di inference le colonne normalizzate sono le stesse del training, ma
> vengono caricate da `mins.csv`/`maxs.csv` (storici) invece di essere ricalcolate.

## 3. Architettura del modello

Da `train_model.ipynb` (cell 18) e `model_summary.txt`:

| Layer | Tipo | Output shape | Parametri |
|---|---|---|---|
| `input_layer` | Input | (None, n_row) | 0 |
| `layerDense1` | Dense, relu | (None, 100) | 4.000 |
| `layerDense2` | Dense, relu | (None, 80) | 8.080 |
| `layerDense3` | Dense, relu | (None, 20) | 1.620 |
| `output_layer` | Dense, **sigmoid** | (None, 1) | 21 |

- `n_row` = numero di feature (dimensione dell'input).
- Total params per il modello di esempio: 41.165; trainable: 13.721.
- Il layer di output è una **sigmoid**, quindi l'output "grezzo" è in `[0,1]`;
  viene poi denormalizzato in `apply_prediction`.

## 4. Compilazione e training

- Ottimizzatore: `Adam(learning_rate=context['learning_rate'])`.
- Loss: `mean_squared_error` (MSE).
- Split: `train_test_split(data, labels, test_size=0.2, random_state=1, shuffle=True)`
  (scikit-learn).
- Training: `model.fit(X, Y, epochs=epochs, shuffle=True, batch_size=batch_size,
  validation_split=validation_split)` con i parametri dal `conf.ini` del modello:
  - `epochs` (es. 30),
  - `batch_size` (es. 128),
  - `validation_split` (es. 0.2),
  - `learning_rate` (es. 0.001).

## 5. Salvataggio

- `model.keras` (formato Keras) nel `portfolio_dir` del modello.
- `model_summary.txt` (output di `model.summary()`).
- `maxs.csv`, `mins.csv`.
- `label_min` / `label_max` in `conf.ini`.

> Nota: il modello salvato è per un **portfolio modello** specifico. Se il portfolio
> di trading usa `model = model` nel proprio `conf.ini`, in fase di inference si
> carica `model.keras` dal `model_dir` (che è il portfolio modello). Se invece il
> portfolio ha id `model_*`, il modello è se stesso.

## 6. Valutazione (nel notebook)

Su `X_test`:

- **Denormalizzazione** delle predizioni: `p * (label_max - label_min) + label_min`.
- **RMSE**: `sqrt(mean((label - p)^2))` — nel notebook chiamato "Mean Squared Error"
  ma è la radice (RMS).
- **Classificazione sul segno**:
  - `tp`: label > 0 e p > 0;
  - `tn`: label ≤ 0 e p ≤ 0;
  - `fp`: label ≤ 0 e p > 0;
  - `fn`: label > 0 e p ≤ 0.
- **Precision** = `tp / (tp + fp)`.
- **Recall** = `tp / (tp + fn)`.
- **Accuracy** = `(tp + tn) / len(X_test)`.
- **Correlazione** (Pearson) tra label e predizioni denormalizzate.

> Non esiste un criterio di "accettazione" formalizzato nel codice: le metriche sono
> solo stampate. La scelta del modello migliore è stata fatta manualmente
> confrontando i modelli `model_*`.

## 7. Inference — `hayai_bo.apply_prediction()`

1. Legge `f001_features.parquet` del **portfolio di trading**.
2. Carica `model.keras` dal `model_dir` (`keras.saving.load_model`).
3. **Tiene solo l'ultima data**: `df = df[df['date'] == df["date"].max()]`.
4. Separa `symbol` e `date` in `df_asset`; droppa `date`, `symbol`, `target`.
5. **Normalizza** con `mins.csv`/`maxs.csv` del modello:
   - `mins = mins.drop('target', errors='ignore')` (idem maxs);
   - `df = (df - mins) / (maxs - mins)`.
6. `x = df.values`; `predictions = model.predict(x, verbose=0)`.
7. **Denormalizza**: `predictions = predictions * (label_max - label_min) + label_min`.
8. **Clip**: `predictions = predictions.clip(clip_min, clip_max)`.
9. Rimette `symbol` e `date`; salva `f002_predictions.parquet`.

Risultato: una riga per asset con la `prediction` per l'ultima data valida.

## 8. Parametri del modello usati a runtime

Da `conf.ini` del **modello** (sezione `[predictions]`):

| Parametro | Default | Uso |
|---|---|---|
| `clip_min` | -5 | Clipping inferiore di predizione e target |
| `clip_max` | 5 | Clipping superiore |
| `label_min` | (obbligatorio) | Min del target (denormalizzazione) |
| `label_max` | (obbligatorio) | Max del target (denormalizzazione) |

> Se `label_min`/`label_max` mancano, `create_context` lancia un errore di lettura
> (`conf_model.getfloat('predictions', 'label_min')` senza fallback).

## 9. Note per la riscrittura

- La **normalizzazione min-max** è salvata su disco: è parte essenziale dell'artefatto
  del modello. In un nuovo sistema conviene unificare modello + parametri in un unico
  artefatto versionato.
- Il layer di output sigmoid + denormalizzazione lineare è un pattern atipico
  (l'output sigmoid viene linearmente rimappato su `[label_min, label_max]`).
- Il target è normalizzato per `vol_20` e clippato: la predizione rappresenta un
  "rendimento atteso normalizzato", non un prezzo.
- La fase di inference non ricalcola min/max: dipende dalla coerenza tra le feature
  del portfolio di trading e quelle del training del modello (stesso ordine di
  colonne, stesse dummies).
