# Hayai V2 — Model Research — Manuale operativo

> **RICERCA CONCLUSA — Settembre 2026**
> Nessuno degli esperimenti EXP-001..EXP-016 ha superato il modello v2 attivo.
> Il segnale cross-sezionale più forte (EXP-005 + macro) ha raggiunto Spearman ≈ +0.016
> sul test set — troppo debole per giustificare un cambio di modello.
> **Il modello `stock_model v2` (24 feature, MLP 100-80-20-1) rimane in produzione.**
> Il codice e i risultati di questa ricerca sono conservati a fini di riferimento storico.

---

## Setup iniziale

```powershell
cd C:\Users\semboli\Documents\progetti\hayai\model_research
..\venv\Scripts\Activate.ps1
pip install pyyaml statsmodels   # pacchetti extra non nel venv principale
```

---

## Struttura della directory

```
model_research/
├── MANUAL.md                   ← questo file
├── requirements.txt            ← dipendenze (no FastAPI, no web)
├── settings.py                 ← costanti globali: path, finestre, soglie
├── run_experiment.py           ← entry point per gli esperimenti
├── config/
│   └── universe.yaml           ← 90 stock + 16 macro inputs
├── data/
│   ├── downloader.py           ← scarica OHLCV da yfinance → cache/
│   ├── validator.py            ← controlla qualità dei dati
│   └── cache/                  ← parquet per simbolo (gitignored)
├── features/                   ← (da implementare)
├── models/                     ← (da implementare)
├── training/                   ← (da implementare)
├── evaluation/                 ← (da implementare)
└── experiments/
    ├── registry.yaml           ← EXP-001 → EXP-016
    └── results/                ← output per esperimento (gitignored)
```

---

## Comandi principali

### Downloader

```powershell
# Scarica tutti i simboli mancanti o scaduti (>1 giorno)
python -m data.downloader

# Forza il re-download di tutto
python -m data.downloader --force

# Singolo simbolo (per test)
python -m data.downloader --symbol NVDA

# Solo stock (no macro)
python -m data.downloader --stocks

# Solo macro (indici, commodities, rates, FX)
python -m data.downloader --macro
```

### Validator

```powershell
# Solo report — nessuna modifica
python -m data.validator

# Rimuove i falliti da universe.yaml
python -m data.validator --fix

# Rimuove anche dalla tabella instrument nel DB di produzione
python -m data.validator --fix --db

# Soglie personalizzate
python -m data.validator --min-rows 750 --min-volume 500000 --max-staleness 20
```

**Soglie di default:**

| Parametro | Default | Significato |
|---|---|---|
| `--min-rows` | 500 | Minimo di trading days (~2 anni) |
| `--max-missing` | 0.05 | Max 5% NaN nel close |
| `--max-staleness` | 30 | Ultimo dato non più vecchio di 30 gg |
| `--min-volume` | 100000 | Volume medio giornaliero minimo |

### Feature engineering

```powershell
# Costruisce il dataset completo e lo salva in data/features/
python -m features.stock_features

# Finestra diversa (per EXP-008/010)
python -m features.stock_features --window 10
python -m features.stock_features --window 40

# Descrive un dataset già salvato senza ricostruirlo
python -m features.stock_features --load
python -m features.stock_features --load --window 10
```

Output in `data/features/`:
- `dataset_w20.npz` — X (n_samples, 20, 12) e y (n_samples, 3) compressi
- `dataset_w20_meta.parquet` — date, ticker, settore per ogni campione

**12 feature per ogni giorno della finestra:**

| Feature | Formula |
|---|---|
| `log_ret_1d` | log(close_t / close_{t-1}) |
| `log_ret_5d` | log(close_t / close_{t-5}) |
| `log_ret_20d` | log(close_t / close_{t-20}) |
| `vol_5d` | std(log_ret, 5) × √252 |
| `vol_20d` | std(log_ret, 20) × √252 |
| `vol_ratio` | vol_5d / vol_20d |
| `open_close` | (open - close) / close |
| `high_close` | (high - close) / close |
| `low_close` | (low - close) / close |
| `high_low` | (high - low) / close |
| `volume_ratio` | volume / volume_ma20 |
| `volume_trend` | volume_ma5 / volume_ma20 |

**3 target (y5, y10, y15):** log return forward normalizzato per vol_20d, clipped a [-3, +3].

### Walk-forward

```powershell
# Mostra la struttura dei fold sul dataset w20
python -m training.walk_forward

# Con timeline ASCII
python -m training.walk_forward --timeline

# Modalità expanding (finestra di training cresce)
python -m training.walk_forward --mode expanding

# Parametri custom
python -m training.walk_forward --train 3 --val 6 --test 6 --embargo 20 --holdout 12
```

**Struttura di ogni fold:**

```
TRAIN (2 anni) | embargo | VAL (6 mesi) | embargo | TEST (6 mesi) | → slide
```

- Embargo = 20 trading days tra ogni split → evita leakage dai target sovrapposti (Y15 usa dati fino a t+15)
- Rolling: finestra di training fissa, scorre in avanti di 6 mesi per fold
- Expanding: training cresce includendo tutto il passato
- **HOLDOUT (ultimi 12 mesi): mai usato per scegliere il modello — solo EXP-016**

### Esperimenti

```powershell
# Lista tutti gli esperimenti con stato
python run_experiment.py --list

# Mostra la config di un esperimento senza eseguirlo
python run_experiment.py --exp EXP-001 --dry-run

# Esegui EXP-001 (baseline momentum — nessun training)
python run_experiment.py --exp EXP-001

# Oppure direttamente il modulo baseline
python -m models.baseline                          # tutte le strategie, Y5
python -m models.baseline --strategy mom_20d       # una sola strategia
python -m models.baseline --horizon 10             # orizzonte Y10
```

**EXP-002/003 — DNN:**

```powershell
python run_experiment.py --exp EXP-002   # linear output
python run_experiment.py --exp EXP-003   # scaled arctan output

# oppure direttamente con parametri custom
python -m models.dnn --activation linear
python -m models.dnn --activation arctan --dropout 0.2 --lr 0.0001
```

Output per fold salvato in `experiments/results/EXP-002/fold_00/model.keras`.

**Nota diagnostica EXP-002:** La configurazione originale con lr=1e-3 causava early stopping a epoch=1 su tutti i 15 fold (Spearman ~+0.009, non stabile). Test diagnostici con lr=1e-4:

| Config | Y5 Spearman | p-value | Monotone |
|---|---|---|---|
| lr=1e-4 solo | +0.007 | 0.019 | 0.50 |
| **lr=1e-4 + dropout=0.2** | **+0.016** | **<0.001** | **0.75** |
| lr=1e-4 + huber loss | +0.013 | <0.001 | 0.75 |

**DEFAULT_LR aggiornato a 1e-4 in settings.py. Tutti i modelli futuri usano questa lr di default.**

**EXP-011/012 — CNN e GRU:**

```powershell
python run_experiment.py --exp EXP-011   # 1D CNN
python run_experiment.py --exp EXP-012   # GRU

# oppure con parametri custom
python -m models.cnn --dropout 0.2 --lr 0.0001
python -m models.gru --dropout 0.2 --recurrent-dropout 0.1 --lr 0.0001
```

**EXP-001 — Strategie baseline:**

| Strategia | Segnale | Note |
|---|---|---|
| `random` | rumore casuale | lower bound — Spearman ≈ 0 |
| `equal_weight` | tutti a 0 | nessun ranking |
| `mom_5d` | log return 5 giorni | momentum breve |
| `mom_20d` | log return 20 giorni | momentum medio |
| `mom_vol_adj` | mom_20d / vol_20d | momentum aggiustato per vol |

Risultati salvati in `experiments/results/EXP-001/`.

---

## Workflow tipico

```
1. Scarica i dati
   python -m data.downloader

2. Valida la qualità
   python -m data.validator

3. Se ci sono falliti, rimuovili
   python -m data.validator --fix

4. (Opzionale) aggiorna anche il DB di produzione
   python -m data.validator --fix --db

5. Costruisci le feature
   python -m features.stock_features     ← da implementare

6. Esegui EXP-001 (baseline momentum)
   python run_experiment.py --exp EXP-001

7. Esegui EXP-002 (DNN baseline)
   python run_experiment.py --exp EXP-002

8. Confronta i risultati
   experiments/results/EXP-001/
   experiments/results/EXP-002/
```

---

## Universo (universe.yaml)

- **90 stock** suddivisi in 5 settori: Technology, AI_Software, Semiconductors, Biotech, Defense
- **16 macro inputs** (non nel ranking): 6 indici, 4 commodities, 3 rate futures, 3 FX
- Ogni stock ha `valid_from`: un ticker non può apparire in training prima di quella data (evita survivorship bias)

Per modificare l'universo: editare `config/universe.yaml` a mano, poi ri-eseguire il validator.

### Gotcha YAML

In YAML 1.1 le parole `ON`, `OFF`, `YES`, `NO`, `TRUE`, `FALSE` senza virgolette vengono parsate come booleani. I ticker problematici vanno quotati:

```yaml
- ticker: "ON"   # ON Semiconductor — DEVE avere le virgolette
```

---

## Parametri globali (settings.py)

| Costante | Valore | Significato |
|---|---|---|
| `DATA_START` | 2015-01-01 | Prima data di download |
| `DEFAULT_WINDOW` | 20 | Trading days per campione |
| `TARGET_HORIZONS` | [5, 10, 15] | Orizzonti forward return (giorni) |
| `TARGET_CLIP` | 3.0 | Clipping del target normalizzato |
| `WF_TRAIN_YEARS` | 2 | Anni di training per fold |
| `WF_VAL_MONTHS` | 6 | Mesi di validation per fold |
| `WF_TEST_MONTHS` | 6 | Mesi di test per fold |
| `WF_EMBARGO_DAYS` | 20 | Gap tra train/val/test (purging) |
| `DEFAULT_SEED` | 42 | Random seed |
| `DEFAULT_LR` | 1e-4 | Learning rate (cambiato da 1e-3 dopo diagnostica EXP-002) |

---

## Sequenza degli esperimenti

| ID | Modello | Obiettivo |
|---|---|---|
| EXP-001 | Baseline momentum | Benchmark non-ML da battere |
| EXP-002 | DNN linear | Baseline DNN |
| EXP-003 | DNN arctan | Confronto activation |
| EXP-004 | DNN solo stock | Ablation: niente macro |
| EXP-005 | DNN + indici | Ablation: aggiungi indici |
| EXP-006 | DNN + indici + commodities | Ablation progressiva |
| EXP-007 | DNN full macro | Tutte le variabili macro |
| EXP-008 | DNN window 10d | Ablation finestra |
| EXP-009 | DNN window 20d | Finestra di riferimento |
| EXP-010 | DNN window 40d | Ablation finestra |
| EXP-011 | 1D CNN | Architettura alternativa |
| EXP-012 | GRU | Architettura recurrente |
| EXP-013 | Miglior arch + target relativo | Target market-relative |
| EXP-014 | Miglior modello + portfolio | Ottimizzazione portafoglio |
| EXP-015 | Walk-forward finale | Validazione completa |
| EXP-016 | Final blind holdout | **Una sola esecuzione** |

> **Regola**: EXP-016 va eseguito una sola volta, alla fine, per il modello scelto. Non usare i risultati per fare ulteriori modifiche.

---

## Conclusione della ricerca (settembre 2026)

**Decisione: nessun modello sostituisce il V2 attuale.**

| Esperimento | Y5 Spearman | Esito |
|---|---|---|
| EXP-001 Baseline momentum | ~0 | Benchmark |
| EXP-002 DNN linear (lr=1e-3) | +0.009 | Non riproducibile (epoch=1) |
| DNN linear lr=1e-4 + dropout=0.2 | +0.016 | Meglio, ma non riproducibile |
| EXP-011 CNN | +0.005 | Non significativo |
| EXP-012 GRU | +0.011 | Significativo solo su Y10 |
| EXP-005 DNN + indici macro | ~0 | Peggiorato |
| EXP-013 DNN target relativo | +0.005 | Non significativo |

**Problemi strutturali:**
1. **Segnale troppo debole**: Spearman +0.016 non copre i costi di rebalancing giornaliero (~25% annuo di drag)
2. **Non-determinismo TF/GPU**: i risultati variano tra run con lo stesso seed
3. **Feature insufficienti**: 12 feature OHLCV non bastano; servirebbero dati fondamentali o alternativi

**Per riprendere la ricerca servirebbero:**
- Feature fondamentali (earnings, revenue growth, P/E, margins)
- Alternative data (sentiment, news flow, satellite)
- Spearman stabile e riproducibile >0.03
- Rebalancing settimanale/mensile per ridurre i costi

---

## Evaluation (evaluation/metrics.py)

```python
from evaluation.metrics import evaluate, print_report, compare_experiments, save_results

# valuta un orizzonte (es. Y5 = indice 0)
results = evaluate(y_pred[:, 0], y_true[:, 0], meta, label="EXP-002 Y5")
print_report(results)

# salva metriche scalari + serie Spearman + tabella quintili
save_results(results, out_dir / "metrics_y5.csv")

# tabella comparativa R10 tra più esperimenti
df = compare_experiments([results_exp001, results_exp002, results_exp003])
```

**Output di `evaluate()`:**

| Metrica | Cosa misura |
|---|---|
| `spearman.mean` | Spearman medio giornaliero (metrica primaria) |
| `spearman.p_value` | Significatività statistica (t-test H0: mean=0) |
| `spearman.ci_lo/hi_95` | Intervallo di confidenza 95% bootstrap |
| `spearman.pct_positive` | % giorni con Spearman > 0 |
| `quintiles.spread` | Rendimento Q5 − rendimento Q1 |
| `quintiles.monotonic` | Frazione coppie con Q(i+1) > Q(i), range [0,1] |
| `prediction.hit_rate` | % previsioni con segno corretto |
| `prediction.mae/rmse/r2` | Qualità predittiva assoluta |
| `sector.*` | Spearman per settore (Technology, Biotech…) |

## Metrica primaria

**Daily Spearman**: ogni giorno ordina le stock per prediction e per actual future return, calcola la correlazione di Spearman. La metrica riportata è la media su tutti i giorni del test set.

Un modello è candidato per Hayai solo se:
1. `mean_daily_spearman > 0` (e statisticamente significativo)
2. `Q5 - Q1 > 0` (i top quintile battono i bottom quintile)
3. Il risultato è stabile in più periodi e regimi di mercato
4. Il rendimento netto (dopo costi) è positivo

---

## Note

- I parquet in `data/cache/` sono gitignored — rigenerabili con `python -m data.downloader`
- I risultati in `experiments/results/` sono gitignored — solo `registry.yaml` è versionato
- Il DB di produzione (`hayai-new/`) è usato in sola lettura (solo `--db` nel validator scrive)
- Lanciare sempre i comandi dalla directory `model_research/` come working directory
