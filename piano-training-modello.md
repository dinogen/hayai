# Piano Operativo: Ingestione Universo di 100 Asset (4-5 Anni) e Training del Modello ML

Questo documento definisce il piano strutturato per estendere l'universo di training a circa **100 asset** (tra azioni, ETF e benchmark/obbligazioni) con uno storico di **4-5 anni**, e per implementare lo script batch di download ed addestramento del modello Keras/ONNX.

---

## Task 1: Definizione e Popolamento dell'Universo di 100 Asset
- **Stato**: todo
- **Scopo**: Selezionare e registrare nel database MariaDB circa 100 strumenti finanziari liquidi e diversificati (S&P 500 top components, ETF settoriali, commodity, bond yields come `^TNX`).
- **Risultato atteso**: Tabella `instrument` e `portfolio_instrument` popolate con ~100 asset attivi.
- **Todolist**:
  - [ ] Compilare la lista dei simboli yfinance (circa 80-90 azioni/ETF chiave + 10-20 indici/bond/forex).
  - [ ] Aggiornare lo script SQL di seed o creare uno script python per inserire i 100 asset nella tabella `instrument`.
  - [ ] Associare gli asset al portfolio di training.

## Task 2: Estensione dello Storico a 4-5 Anni (Ingestion Storica)
- **Stato**: todo
- **Scopo**: Scaricare i dati OHLCV giornalieri da yfinance per tutti i 100 asset coprendo un arco temporale di 4-5 anni (es. da gennaio 2021 a oggi).
- **Risultato atteso**: Tabella `price_daily` popolata con tutti i dati storici giornalieri necessari per il calcolo delle feature di lungo periodo.
- **Todolist**:
  - [ ] Modificare il job di ingestion dati (`app/jobs/data.py` o creare uno script specifico `train_data_ingest.py`) per accettare un parametro `period="5y"`.
  - [ ] Gestire i rate limit di yfinance con delay adeguati tra una chiamata e l'altra.
  - [ ] Eseguire il popolamento dei dati ed effettuare un controllo di integrità sul numero di barre salvate.

## Task 3: Feature Engineering e Generazione del Dataset di Training
- **Stato**: todo
- **Scopo**: Calcolare le feature tecniche type-agnostic (`log_return`, momentum, volatilità, z-score, ecc.) e il target forward per l'intero dataset di 4-5 anni.
- **Risultato atteso**: Un DataFrame o file CSV/parquet contenente tutte le feature e il target pulito da NaN e infiniti.
- **Todolist**:
  - [ ] Implementare la funzione di calcolo feature multi-asset (coerente con `doc-new-app/03-ml-pipeline.md`).
  - [ ] Calcolare il target normalizzato e rimosso di valori estremi (`clip`).
  - [ ] Esportare il dataset pulito in formato parquet o CSV per il training in Jupyter/Python.

## Task 4: Script di Addestramento Keras ed Esportazione in ONNX
- **Stato**: todo
- **Scopo**: Addestrare la rete neurale MLP (`100 → 80 → 20 → 1`) sul dataset di 100 asset storicizzati e convertire il modello in formato ONNX.
- **Risultato atteso**: File `model.onnx`, `metadata.json`, `config.json` e i file di normalizzazione (`mins.csv`, `maxs.csv`) salvati nella cartella dei modelli.
- **Todolist**:
  - [ ] Scrivere lo script python di training (o notebook Jupyter) con `train_test_split`.
  - [ ] Eseguire l'addestramento con Keras (Optimizer Adam, MSE loss).
  - [ ] Esportare il modello in formato ONNX e salvare i parametri di normalizzazione.
  - [ ] Registrare il nuovo modello nella tabella `model_registry` del database.

## Task 5: Validazione e Test dell'Inferenza Batch
- **Stato**: todo
- **Scopo**: Verificare che il job di predizione (`job predict`) sul Raspberry Pi o in ambiente locale legga correttamente il modello ONNX e produca i `quant_score` attesi per l'universo.
- **Risultato atteso**: Inferenza completata senza errori e salvataggio dei punteggi in `model_prediction`.
- **Todolist**:
  - [ ] Eseguire il comando di predizione sul set di test.
  - [ ] Verificare che non vi siano disallineamenti tra le feature di training e di inferenza.
  - [ ] Testare l'esecuzione tramite lo script batch complessivo.
