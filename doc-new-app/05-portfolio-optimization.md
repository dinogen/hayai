# 05 — Ottimizzazione Portafoglio & Composizione Long/Short (€5.000 Experiment)

Questo documento descrive la logica di allocazione del portafoglio basata sul
**segnale ibrido** (`final_signal`), tarata specificamente per l'esperimento
personale con un capitale di **€5.000**.

---

## 1. Regole di Allocazione e Pesi (`job recommend`)

Ogni notte, dopo il calcolo del segnale ibrido in `portfolio_signal`, il batch calcola la composizione target:

1. **Peso Grezzo (`weight_raw`)**: proporzionale a `final_signal / vol_20`.
2. **Selezione Top/Bottom**:
   - I migliori **`n_long`** strumenti (es. 5) con segnale positivo.
   - I peggiori **`n_short`** strumenti (es. 3) con segnale negativo.
3. **Normalizzazione**: la somma dei valori assoluti dei pesi è esattamente **1.0**.

---

## 2. Dimensionamento per il Capitale di €5.000

Poiché il portafoglio ha un capitale iniziale di **€5.000** e un livello di rischio `risk_percentage` impostato al 90% (0.90):
- **Capitale Investito Totale**: $€5.000 \times 0.90 = \mathbf{€4.500}$ (il 10%, ovvero €500, resta in liquidità di sicurezza o buffer).
- **Importo Target per Strumento (`target_amount`)**:
  $$\text{target\_amount} = \text{weight} \times €4.500$$
- **Quantità di Quote (`target_qty`)**:
  $$\text{target\_qty} = \text{round}\left(\frac{\text{target\_amount}}{\text{prezzo corrente}}\right)$$

### Esempio Pratico per la Webapp:
- **AAPL** (Long, peso 15%):
  - Importo allocato: $0.15 \times €4.500 = \mathbf{€675}$.
  - Se AAPL quota $225, la quantità indicativa è $\text{round}(675 / 225) = \mathbf{3 \text{ quote}}$.

Questo ti permette di aprire la webapp il **martedì**, vedere esattamente quanti pezzi detenere o muovere per ciascun asset e discuterne con il promotore.
