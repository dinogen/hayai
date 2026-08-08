# 03 — Pipeline ML & Signal Hybrid (Quant + LLM)

Questo documento descrive il ciclo di vita del machine learning in HAYAI v2:
1. Il **training periodico** (ogni 2-3 mesi) in Jupyter su PC.
2. L'esportazione in **ONNX** e il deploy sul Raspberry Pi.
3. L'**inferenza giornaliera** e l'**aggiustamento ibrido del segnale** tramite DeepSeek.

---

## 1. Ciclo di Vita del Modello (Training Periodico)

A differenza dei dati (aggiornati quotidianamente), i modelli Keras non hanno bisogno di essere addestrati ogni giorno. La cadenza ottimale è **ogni 2 o 3 mesi**, catturando i cambiamenti strutturali di medio termine nei mercati.

```
[ PC Sviluppo ]
 Storico DB ──► Notebook Jupyter ──► Addestramento Keras (Asset Class Model)
                                           │
                                           ▼
                                Export in ONNX + mins/maxs.csv + config.json
                                           │
                                           ▼
 [ Raspberry Pi ] ◄── rsync / deploy ──────┘ (Registrato in model_registry)
```

### 1.1 Modelli Specializzati per Asset Class
Visto che le dinamiche di prezzo differiscono radicalmente tra azioni, valute e obbligazioni, la nuova architettura supporta modelli separati registrati in `model_registry`:
- **`stock_model`**: Azioni ed ETF (può includere feature di volatilità e range).
- **`fx_model`**: Coppie di valute (`EURUSD=X`, ecc.).
- **`bond_model`**: Rendimenti obbligazionari e ETF bond (`^TNX`, `BND`, ecc.).

---

## 2. Architettura del Modello e Feature (Type-Agnostic)

L'architettura della rete neurale densa (MLP) riprende la struttura collaudata del legacy (`train_model.ipynb`, doc `04`), con strati densi `100 → 80 → 20 → 1 (sigmoid)` e funzione di loss `Mean Squared Error`.

### Feature Set Comune
Per evitare la complessità delle dummies geografiche del legacy (doc `03`), si utilizzano feature tecniche calcolate per singolo strumento:
- `log_return`: Rendimento logaritmico a 5 giorni.
- `mom_5`, `mom_10`, `mom_20`: Momentum a 5, 10, 20 giorni.
- `vol_10`, `vol_20`: Volatilità a 10 e 20 giorni.
- `vol_ratio`: Rapporto `vol_10 / vol_20`.
- `zscore_20`: Distanza normalizzata dalla media mobile a 20 giorni.
- `trend_50`: Distanza dalla media mobile a 50 giorni.
- `vol_regime`: Rapporto volatilità a 10 e 60 giorni.
- `mom_vol_adj`: Momentum 20 giorni diviso volatilità 20 giorni (`mom_20 / vol_20`).
- **Target**: `clip(ln(close_{t+trd} / close_t) / vol_20, clip_min, clip_max)`.

---

## 3. Inferenza Giornaliera sul Raspberry Pi (`job predict`)

Ogni notte, il batch sul Raspberry Pi:
1. Legge dal database l'ultima barra giornaliera disponibile per gli strumenti del portafoglio.
2. Calcola le feature tecniche correnti.
3. Normalizza i dati usando i file `mins.csv` e `maxs.csv` associati al modello attivo (`model_registry`).
4. Esegue l'inferenza tramite **`onnxruntime`** (leggerissimo su CPU ARM, senza bisogno di installare TensorFlow sul Pi).
5. Denormalizza l'output: `quant_score = output * (label_max - label_min) + label_min`.
6. Salva il risultato in `model_prediction`.

---

## 4. Signal Adjustment (Integrazione con DeepSeek)

Il `quant_score` calcolato da Keras è puramente matematico e ignora eventi esogeni improvvisi (es. scandali aziendali, notizie geopolitiche, cambi di tassi imprevisti). 

Qui entra in gioco **DeepSeek API**:

### 4.1 Come funziona l'aggiustamento del segnale
1. Il job `news` scarica le notizie del giorno per lo strumento.
2. Il modulo LLM invia le notizie a DeepSeek con un prompt strutturato (vedi `04-news-llm-pipeline.md`).
3. DeepSeek restituisce un JSON contenente:
   - `sentiment`: `'bullish'`, `'neutral'`, `'bearish'`
   - `confidence`: valore da `0.0` a `1.0`
   - `rationale`: spiegazione testuale in italiano.
4. Il batch calcola il **Modificatore di Sentiment** (`llm_sentiment_modifier`), un valore compreso tra `-0.20` e `+0.20` ricavato da sentiment e confidence.
5. **Segnale Finale Ibrido**:
   $$\text{final\_signal} = \text{quant\_score} + \text{llm\_sentiment\_modifier}$$
6. Il `final_signal` viene salvato in `portfolio_signal`, accompagnato dalla tesi di investimento (`ai_rationale`).

### Esempio pratico:
- **AAPL**: `quant_score` = `+0.80` (forte segnale tecnico rialzista).
- **DeepSeek Sentiment**: `bullish` (confidence 0.90) a causa di vendite record di iPhone. `llm_sentiment_modifier` = `+0.15`.
- **`final_signal`** = $0.80 + 0.15 = \mathbf{0.95}$.
- Se invece le notizie fossero state pessime (`bearish`), il modificatore avrebbe potuto smorzare o invertire il segnale, proteggendo il portafoglio.
