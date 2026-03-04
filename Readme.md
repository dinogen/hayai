# Hayai – Pipeline di Trading

## 1. Ingestion

Acquisisce i dati OHLCV, calcola le features e le labels, e produce un dataset aggiornato.

**Output:** `features.parquet` (`symbol`, `open`, `high`, `low`, `close`, `volume`, tutte le features, `target`)

---

## 2. Training

Utilizza il dataset aggiornato `features.parquet` per il training del modello.

**Output:** `model.keras`

---

## 3. Prediction

Usa il modello per calcolare il **signal** e i **pesi** (`signal / vol_20`) per tutto il portafoglio.

- Prende i primi 5 e gli ultimi 5 per peso
- Calcola 10 position che devono avere somma = 1

**Output:** `weights.parquet` (`symbol`, `weight_new`)

---

## 4. Position New

Costruisce la nuova posizione target unendo posizione attuale e pesi previsti.

1. Prende la lista della posizione attuale (`symbol`, `qty`) da Alpaca
2. Riprende la lista `weights.parquet`
3. Esegue il merge ottenendo la lista completa con `qty_old` e `weight_new` per ogni asset (0.0 – 1.0)

**Output:** `position_new.parquet` (`symbol`, `qty_old`, `weight_new`)

---

## 5. Define new Qty

Converte i pesi in quantità e calcola le differenze rispetto alla posizione attuale.

1. Prende i prezzi trade per i symbol di `position_new.parquet`
2. Trasforma `weight_new` in `qty_new` usando il prezzo trade
3. Calcola `qty_diff = qty_new - qty_old`
4. Calcola la differenza percentuale rispetto a `qty_old` (0 – 1)
5. Rimuove le righe con variazione inferiore al 20% (< 0.2)

**Output:** `position_new_qty.parquet` (`symbol`, `qty_old`, `qty_new`, `qty_diff`, `price_trade`)

---

## 6. Execution

Scorre `execution.parquet` e genera gli ordini secondo la seguente logica:

| Condizione | Azione |
|---|---|
| `qty_new = 0` | **Chiudi** la posizione |
| `qty_old = 0` e `qty_new > 0` | **BUY** `qty_new` |
| `qty_old = 0` e `qty_new < 0` | **SELL** `round(qty_new)` |

### Posizione long esistente (`qty_old > 0`)

| `qty_new` | `qty_diff` | Azione | Esempio |
|---|---|---|---|
| `> 0` | `> 0` | **BUY** `qty_diff` | +100 → +150 ⇒ compro 50 |
| `> 0` | `< 0` | **SELL** `round(qty_diff)` | +150 → +100 ⇒ vendo 50 |
| `< 0` | — | Chiudi posizione, **SELL** `round(qty_new)` | Inversione long → short |

### Posizione short esistente (`qty_old < 0`)

| `qty_new` | `qty_diff` | Azione | Esempio |
|---|---|---|---|
| `< 0` | `> 0` | **BUY** `qty_diff` | −150 → −100 ⇒ compro 50 |
| `< 0` | `< 0` | **SELL** `round(qty_diff)` | −100 → −150 ⇒ vendo 50 |
| `> 0` | — | Chiudi posizione, **BUY** `qty_new` | Inversione short → long |
