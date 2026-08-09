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

### Feature Set Comune (v2)
Il modello `stock_model v2` usa 24 feature: le 12 base più 12 feature **cross-sezionali** e di
**regime di mercato**, calcolate sul pannello completo `(simbolo, data)`. Queste sono allineate
all'uso reale del modello (selezione top long / bottom short), che il backtest ha mostrato essere
il punto debole di v1 (Spearman ~0.03).

Feature base (per singolo strumento):
- `log_return`: Rendimento logaritmico a 5 giorni.
- `mom_5`, `mom_10`, `mom_20`: Momentum a 5, 10, 20 giorni.
- `vol_10`, `vol_20`: Volatilità a 10 e 20 giorni.
- `vol_ratio`: Rapporto `vol_10 / vol_20`.
- `zscore_20`: Distanza normalizzata dalla media mobile a 20 giorni.
- `trend_50`: Distanza dalla media mobile a 50 giorni.
- `vol_regime`: Rapporto volatilità a 10 e 60 giorni.
- `mom_vol_adj`: Momentum 20 giorni diviso volatilità 20 giorni (`mom_20 / vol_20`).
- `volume_shock`: Volume diviso media mobile 20 giorni del volume.

Feature cross-sezionali (relative all'universo, stesse date):
- `ret_1`: Rendimento logaritmico a 1 giorno.
- `x_rank_mom5`, `x_rank_mom20`, `x_rank_trend50`: Percentile rank della feature dentro l'universo alla stessa data.
- `rel_mom5_spy`, `rel_mom20_spy`: Momentum relativo vs SPY (`mom_i − mom_SPY`).
- `excess_ret_5`: Rendimento 5gg in eccesso rispetto alla media dell'universo.
- `beta_20`: Beta rolling a 20 giorni vs SPY.

Feature di regime di mercato (condivise, valore uguale per tutti gli strumenti nella data):
- `mkt_ret_5`, `mkt_ret_20`: Rendimenti SPY a 5 e 20 giorni.
- `breadth_20`: Frazione di strumenti dell'universo con `mom_20 > 0` (ampiezza del mercato).
- `dispersion_20`: Deviazione standard cross-sezionale di `mom_20`.

**Preprocessing**: prima del min-max, ogni feature viene **winsorizzata** ai percentili
0.5/99.5 (gli outlier tipo `mom_vol_adj` comprimono la scala del min-max). Il target resta
`clip(ln(close_{t+trd} / close_t) / vol_20, clip_min, clip_max)`.

**Training**: MLP `100 → 80 → 20 → 1 (sigmoid)`, loss MSE, ma con **early stopping**
(`patience=5`, `restore_best_weights=True`) fino a 50 epochs (v1 usava 10 epochs fisse, sotto-addestrato).

### Feature testate e scartate (esperimento v3)
Sono state testate altre due feature (modello `v3`): `dow_sin`/`dow_cos` (giorno della settimana
in encoding ciclico) e `days_since_high` (giorni di trading dall'ultimo massimo di chiusura a 252
giorni, `log1p`). Rispetto a v2 il verify migliorava di poco (R² 0.171 vs 0.163) ma il backtest
**peggiorava** (Spearman 0.133 vs 0.174; spread cumulato +2.69 vs +3.24; hit-rate short 52.6% vs
58.3%). Per come il modello è usato (ranking long/short) le due feature non aggiungono valore:
**v2 (24 feature) resta il modello di produzione**; v1 e v3 sono archiviati in `model_registry`.

---

## 3. Inferenza Giornaliera sul Raspberry Pi (`job predict`)

Ogni notte, il batch sul Raspberry Pi:
1. Legge dal database l'ultima barra giornaliera disponibile per **tutti** gli strumenti del portafoglio.
2. Calcola le feature sul **pannello completo** (`compute_panel_features`): le feature
   cross-sezionali e di regime richiedono l'universo alla stessa data, non il singolo strumento.
3. Prende l'ultima riga per strumento e normalizza usando i file `mins.csv` e `maxs.csv`
   associati al modello attivo (`model_registry`).
4. Esegue l'inferenza **in batch** tramite **`onnxruntime`** (leggerissimo su CPU ARM, senza bisogno di installare TensorFlow sul Pi).
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

---

## 5. Verifica del Modello (`job verify`)

Uno script standalone (`app/jobs/verify_model.py`, registrato come job `verify`) valuta il modello
deployato sul dataset attuale senza riaddestrare. Va lanciato manualmente dopo un retraining o
quando si sospetta un drift dei dati.

```
python -m app.cli verify            # (oppure: python -m app.jobs.verify_model)
python -m app.cli verify --version v4   # valuta un modello specifico (es. holdout v4)
```

Se il `config.json` del modello contiene `split: "time"` (con `train_end`/`val_end`), verify e
backtest riproducono lo split cronologico e valutano **solo sul test holdout** (mai visto
dall'early stopping), fittando min/max e label solo sul train.

La verifica produce un report testuale in `logs/model_verification_<nome>_<versione>_<data>.txt`
(in italiano, con le definizioni delle metriche) e logga in console + `hayai.log`. Controlla:

1. **Assenza di null/NaN**: log di righe raw e dopo `dropna`, con assert di 0 NaN/Inf su
   `X_train`, `X_test`, `y_train`, `y_test` e sulle 100 righe campionate.
2. **Split 80/20**: dimensioni e percentuali effettive di training e test
   (`train_test_split(test_size=0.2, random_state=42)`).
3. **Predizione su tutto il test set** con il modello ONNX deployato, denormalizzazione e clip
   (stessa matematica del job `predict`).
4. **Metriche di regressione** (il target è continuo, quindi non si usano accuracy/precision/recall):
   - **RMSE** (scarto quadratico medio) = `sqrt(media((pred − actual)²))` — errore tipico in unità del target.
   - **MAE** (errore medio assoluto) = `media(|pred − actual|)`.
   - **R²** = `1 − SS_res/SS_tot` — quota di varianza spiegata (per i rendimenti finanziari valori bassi o ~0 sono normali).
   - **Hit-rate direzionale** = % di osservazioni con `sign(pred) == sign(actual)` (direzione su/giù).
   - Baseline ingenua ("predici sempre la media") per contestualizzare RMSE/MAE.
5. **Spot check**: 100 righe casuali del test set (`random_state=42`) con tabella
   `symbol | trade_date | actual | prediction | match`.
6. **Drift min/max**: avvisa se i min/max ricomputati sul dataset attuale differiscono da quelli
   degli artifact del modello deployato (possibile cambio di distribuzione → valutare retraining).

Nota: la normalizzazione dei dati per la verifica usa i min/max ricomputati come nel training,
non quelli degli artifact; eventuali differenze vengono segnalate come potenziale drift.

---

## 6. Backtest della Selezione Long/Short (`job backtest`)

Lo script `app/jobs/backtest_selection.py` (job `backtest`) valuta se il ranking del
`quant_score` separa davvero vincenti e perdenti, usando solo il **test set** del modello
(out-of-sample, 20% split `random_state=42`):

```
python -m app.cli backtest            # (oppure: python -m app.jobs.backtest_selection)
```

Metodologia:
- Ricostruisce il pannello `(symbol, trade_date)` con `quant_score` (inferenza ONNX) e
  ritorno forward a 5 giorni `fwd_log_ret = target * vol_20` (niente look-ahead: solo righe
  con target noto).
- Per ogni data di ribilanciamento: **LONG** top-5 e **SHORT** bottom-5 per `quant_score`,
  confrontati con universo pari-peso e SPY.
- Statistiche su tutte le date (sovrapposte, indicative) e su date **non sovrapposte**
  (ogni 5 giorni di trading) per il P&L cumulato.
- Correlazione cross-sezionale **Spearman** media (`quant_score` vs ritorno).

Metriche nel report (`logs/model_backtest_stock_model_v1_*.txt`): ritorno medio long/short/
universe/SPY, spread long-short, hit-rate (long>0, short<0, long>universe), Spearman medio e
spread cumulato non sovrapposto.

Risultati di riferimento — **v1** (12 feature base, 10 epochs; 228 ribilanciamenti non sovrapposti
su ~5 anni di test): spread long-short medio **+0.46%** per 5 giorni, long +0.36% vs SPY +0.31% vs
universo +0.12%, Spearman medio **+0.03**, hit-rate short<0 48%. Edge cross-sezionale quasi nullo.

Risultati di riferimento — **v2** (24 feature incl. cross-sezionali/regime, winsorize, early
stopping; stesso test set): spread long-short medio **+1.42%** per 5 giorni, long **+0.80%** vs
SPY +0.31% vs universo +0.11%, Spearman medio **+0.17** (6x), hit-rate long>0 66%, short<0 58%,
long>universe 68%, spread cumulato **+3.24** log vs +1.06. Verifica: RMSE 1.145 (da 1.249), R²
**0.163** (da 0.004), hit-rate direzionale **62.9%** (da 54.0%).

**Caveat metodologico (importante)**: il "test set" usato nelle sezioni 5-6 è anche la
`validation_data` dell'early stopping, quindi i numeri di v2 (R² 0.163, Spearman 0.174) sono
**ottimistici**: le epochs sono state scelte sul validation loss. Il confronto v1 vs v2 resta
valido in termini relativi, ma non come stima assoluta della performance.

**Holdout cronologico (v4)** — per avere una stima non contaminata è stato addestrato `v4` con
split per data **70/15/15** (train ≤ 2025-02-28, val ≤ 2025-11-12, test dal 2025-11-13), con
scaler e label fittati solo sul train e test **mai visto dall'early stopping** (registrato come
`draft`, non attivo). Risultati sul holdout:
- Verify: R² **≈ 0** (−0.004), RMSE ≈ baseline (il modello non supera "predici la media"),
  hit-rate direzionale 53.8%.
- Backtest (36 ribilanciamenti non sovrapposti): Spearman **+0.077**, LONG +0.57% vs SPY +0.36%
  vs universo +0.39% per 5gg, SHORT ≈ rumore (hit-rate 50%), spread cumulato +0.41.

**Interpretazione**: l'edge cross-sezionale reale è molto più debole di quanto suggerissero i
numeri random-split. Resta un **tilt long-only** del top-5 che batte leggermente SPY (il lato
più difendibile), mentre il lato short e l'RMSE non hanno valore predittivo robusto. La parte
quant va quindi trattata come input debole dell'ibrido, con il modificatore LLM dominante.

**Esperimento v3**: l'aggiunta di `dow_sin`/`dow_cos` e `days_since_high` non ha aiutato la
selezione (Spearman 0.133 vs 0.174 sul random split). **v2 resta il modello attivo e di
produzione**.
