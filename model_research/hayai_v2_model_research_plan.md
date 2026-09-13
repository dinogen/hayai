# Hayai V2 — Piano di ricerca e sperimentazione del nuovo modello

## 1. Obiettivo

Costruire un nuovo modello quantitativo per Hayai capace di prevedere il comportamento futuro delle azioni appartenenti a un universo selezionato di:

- Technology
- Artificial Intelligence
- Semiconductors
- Biotech
- Defense / Aerospace

Il modello **non traduce in portafoglio indici, commodities, valute o futures**. Questi strumenti sono esclusivamente variabili esplicative utilizzate dalla rete neurale.

L'obiettivo non è ottenere la minima loss possibile, ma verificare sperimentalmente se il modello possiede una capacità predittiva economicamente sfruttabile.

La domanda finale è:

> Il modello riesce a ordinare le azioni in modo tale che quelle con previsione migliore abbiano, statisticamente e stabilmente, rendimenti futuri migliori?

---

# 2. Principio fondamentale

Il progetto deve essere costruito come un **esperimento scientifico**, non come un unico tentativo di machine learning.

Ogni modello deve essere confrontato usando:

- stesso universo;
- stesso periodo;
- stesso dataset;
- stesso periodo out-of-sample;
- stesso metodo di esecuzione;
- stessi costi di trading;
- stessi vincoli di portafoglio.

Ogni modifica deve essere identificabile come un esperimento.

---

# 3. Universo delle azioni

## 3.1 Dimensione

| Dimensione | Numero |
|---|---:|
| Minimo | 60 |
| Target | 100 |
| Massimo | 150 |

Con meno di 60 titoli il ranking cross-sectional diventa troppo dipendente dai singoli titoli. Oltre 150, nella prima fase, il numero di input e il rischio di overfitting aumentano senza un beneficio necessariamente dimostrato.

## 3.2 Composizione indicativa

| Categoria | Target |
|---|---:|
| Technology | 25 |
| Semiconductors | 20 |
| AI / Software | 20 |
| Biotech | 20 |
| Defense / Aerospace | 15 |
| **Totale** | **100** |

Ogni titolo deve avere almeno:

```text
ticker
company
sector
industry
exchange
currency
country
valid_from
valid_to
```

## 3.3 Survivorship bias

Non bisogna costruire oggi un universo di 100 aziende e simulare il passato come se quelle 100 fossero sempre esistite.

Un titolo entrato nell'universo nel 2024 non deve comparire nel training del 2018.

La composizione dell'universo deve quindi essere storicamente ricostruibile.

---

# 4. Strumenti macro utilizzati come input

Indici, commodities, futures, tassi e valute **non appartengono al portafoglio**. Sono esclusivamente input della DNN.

## 4.1 Indici

Set iniziale:

```text
^GSPC     S&P 500
^IXIC     NASDAQ Composite
^NDX      NASDAQ 100
^RUT      Russell 2000
^DJI      Dow Jones Industrial Average
^VIX      CBOE Volatility Index
```

Estensione successiva:

```text
^FTSE     FTSE 100
^GDAXI    DAX
^FCHI     CAC 40
```

## 4.2 Commodities / futures

Core:

```text
CL=F      Crude Oil
HG=F      Copper
GC=F      Gold
SI=F      Silver
```

Estensione:

```text
PL=F      Platinum
ALI=F     Aluminum
NG=F      Natural Gas
```

## 4.3 Futures sui tassi

```text
ZN=F      10-Year Treasury Note Futures
ZF=F      5-Year Treasury Note Futures
ZT=F      2-Year Treasury Note Futures
```

In alternativa o successivamente potranno essere utilizzati direttamente i Treasury yields.

## 4.4 Valute

Core:

```text
EURUSD=X
JPY=X
GBPUSD=X
```

Possibile estensione:

```text
DXY
USD/JPY
EUR/USD
GBP/USD
```

## 4.5 Set iniziale

La prima versione dovrebbe utilizzare circa:

```text
6 indici
+
4 commodities
+
3 rate/futures
+
3 FX
```

circa **16 serie macro**.

Non aggiungere subito decine di variabili macro: prima bisogna dimostrare che aggiungono informazione.

---

# 5. Input delle azioni

Per ogni stock:

```text
Open
High
Low
Close
Volume
```

Non utilizzare direttamente i prezzi assoluti.

Preferire quantità relative, ad esempio:

```text
return_1
return_2
return_5
return_10
open/close
high/close
low/close
volume / average_volume
```

L'obiettivo è far apprendere alla rete la dinamica, non il livello nominale del prezzo.

---

# 6. Finestra temporale

La finestra base sarà:

```text
20 trading days
```

Per ogni stock:

```text
t-19
t-18
...
t-1
t
```

Ogni giorno contiene:

```text
stock features
+
macro features
```

La rete riceve quindi una sequenza temporale.

La dimensione della finestra sarà successivamente oggetto di esperimento:

```text
10
20
40
60
```

---

# 7. Target

Il modello avrà tre target:

```text
Y5
Y10
Y15
```

dove:

```text
Y5  = rendimento futuro a 5 giorni
Y10 = rendimento futuro a 10 giorni
Y15 = rendimento futuro a 15 giorni
```

Preferibilmente usare un rendimento logaritmico normalizzato per la volatilità:

```text
Y5  = log(C[t+5]  / C[t]) / volatility
Y10 = log(C[t+10] / C[t]) / volatility
Y15 = log(C[t+15] / C[t]) / volatility
```

## 7.1 Target assoluto e relativo

Devono essere confrontate due formulazioni.

### A — Absolute return

```text
Y = future_return(stock)
```

### B — Relative return

```text
Y = future_return(stock)
    - future_return(market/universe)
```

Il target relativo è particolarmente interessante perché Hayai deve soprattutto stabilire quali titoli faranno meglio degli altri.

---

# 8. Multitask learning

La rete principale sarà multitask:

```text
                 ┌── Y5
                 │
20-day input ─── DNN
                 │
                 ├── Y10
                 │
                 └── Y15
```

La parte iniziale della rete impara informazioni comuni; i tre output imparano i diversi orizzonti.

La relazione tra Y5, Y10 e Y15 sarà analizzata anche come struttura temporale del segnale.

---

# 9. Architettura baseline

Prima rete:

```text
Input
  ↓
Dense 128
ReLU
  ↓
Dense 64
ReLU
  ↓
Dense 32
ReLU
  ↓
Output 3
```

Due activation finali saranno confrontate.

### Linear

```text
output = x
```

### Scaled Arctan

```text
output = 3 * (2/pi) * atan(x)
```

che limita l'output a:

```text
[-3,+3]
```

La scelta finale deve essere sperimentale.

---

# 10. Modelli da confrontare

## M0 — Baseline non-ML

- momentum 5 giorni;
- momentum 20 giorni;
- momentum + volatility;
- equal weight;
- random ranking.

## M1 — DNN

```text
20 giorni → flatten → Dense → Dense → Dense → 3 output
```

## M2 — DNN + Arctan

Identico a M1, con output Scaled Arctan.

## M3 — 1D CNN

```text
20 giorni
↓
Conv1D
↓
Conv1D
↓
Dense
↓
Y5/Y10/Y15
```

## M4 — GRU

```text
20 giorni
↓
GRU
↓
Dense
↓
Y5/Y10/Y15
```

## M5 — LSTM

Da testare dopo GRU.

## M6 — Hybrid

Eventualmente:

```text
CNN
+
macro branch
+
stock branch
↓
Dense
↓
Y5/Y10/Y15
```

Transformer solo in una fase successiva, se i modelli più semplici mostrano capacità predittiva.

---

# 11. Normalizzazione

La normalizzazione deve essere calcolata **solo sul training set**.

Mai usare train + validation + test per calcolare min/max, mean/std o altri parametri di preprocessing.

---

# 12. Split temporale

**Mai random split.**

Schema:

```text
TRAIN              VALIDATION          TEST
───────────────────┼───────────────────┼──────────────►
                    tempo
```

Preferibilmente utilizzare anche **walk-forward validation**.

---

# 13. Purging / embargo

Poiché Y5, Y10 e Y15 sono finestre sovrapposte, campioni vicini temporalmente non sono completamente indipendenti.

Un campione a `t` con target Y15 utilizza informazioni fino a `t+15`.

Train, validation e test devono quindi essere separati con un gap temporale sufficiente.

Questo evita leakage indiretto tra campioni.

---

# 14. Loss

Confrontare almeno:

```text
MSE
Huber Loss
```

La Huber è interessante per la presenza di outlier nei rendimenti finanziari.

---

# 15. Metriche del modello

## Prediction quality

```text
MAE
RMSE
R²
```

## Ranking quality

```text
Daily Spearman
Daily Kendall
```

## Portfolio relevance

```text
Q5 return
Q1 return
Q5 - Q1
Hit rate
Long-only return
Long-short return
```

---

# 16. Test Spearman

Ogni giorno ordinare le azioni secondo prediction e actual future return.

Calcolare:

```text
Spearman(prediction, actual)
```

e poi:

```text
mean
median
std
percentile 25
percentile 75
```

La metrica principale sarà:

```text
mean_daily_spearman
```

non una singola correlazione calcolata sull'intero dataset.

---

# 17. Quintile test

Ogni giorno ordinare le azioni secondo la prediction:

```text
Q1 = bottom 20%
Q2
Q3
Q4
Q5 = top 20%
```

Calcolare il rendimento futuro di ciascun gruppo.

Il comportamento ideale è:

```text
Q1 < Q2 < Q3 < Q4 < Q5
```

con uno spread Q5-Q1 positivo.

---

# 18. Test long-short

Strategia:

```text
Long  = Q5
Short = Q1
```

Calcolare:

```text
Q5 - Q1
```

È un test particolarmente utile perché riduce l'effetto del mercato generale.

---

# 19. Test long-only

Dato che il portafoglio Hayai è basato sulle azioni:

```text
Long Q5
```

confrontare con:

```text
SPY
QQQ
equal-weight universe
```

---

# 20. Test dei tre orizzonti

Per ogni modello analizzare separatamente:

```text
Y5
Y10
Y15
```

e poi la struttura:

```text
Y5
Y10
Y15
```

Esempio:

```text
Y5  +2.1
Y10 +3.7
Y15 +5.2
```

segnale crescente.

Oppure:

```text
Y5  +3.1
Y10 +1.2
Y15 -0.4
```

segnale di breve periodo in deterioramento.

---

# 21. Horizon consistency test

Misurare:

```text
corr(Y5, Y10)
corr(Y5, Y15)
corr(Y10, Y15)
```

Se i tre output risultano quasi identici, il multitask potrebbe aggiungere poca informazione.

---

# 22. Ablation test

Partire dal modello completo e rimuovere gruppi di input.

### A

Solo stock:

```text
OHLCV
```

### B

Stock + indici

### C

Stock + commodities

### D

Stock + FX

### E

Stock + rates

### F

Tutto

Obiettivo:

> verificare se le variabili macro aggiungono realmente valore.

---

# 23. Test specifico delle commodities

Confrontare:

```text
Base
+
Oil
+
Copper
+
Gold
+
Oil + Copper + Gold
```

per verificare l'informazione incrementale di ciascuna variabile.

---

# 24. Test della dimensione dell'universo

Confrontare:

```text
60
100
150
```

mantenendo identiche le altre condizioni.

Misurare:

- Spearman;
- Q5-Q1;
- stabilità;
- turnover;
- performance;
- diversificazione.

---

# 25. Test per settore

Calcolare le metriche separatamente:

```text
Technology
AI
Semiconductors
Biotech
Defense
```

Questo permette di scoprire se il modello funziona meglio in determinati settori.

---

# 26. Test per regime di mercato

Classificare il mercato in:

```text
Bull
Bear
Sideways
High volatility
Low volatility
```

e calcolare le performance separatamente.

---

# 27. Test della direzione del segnale

Esperimento obbligatorio:

```text
prediction
```

contro:

```text
-prediction
```

Se il segnale invertito funziona meglio, il modello potrebbe aver imparato una relazione inversa.

---

# 28. Test della stabilità temporale

Dividere il periodo di test in più sotto-periodi e verificare che il modello non funzioni soltanto in una finestra specifica.

---

# 29. Test di robustezza

Cambiare:

- seed;
- learning rate;
- batch size;
- numero neuroni;
- dropout;
- loss;
- normalizzazione.

Un modello robusto non dovrebbe cambiare completamente comportamento al variare del seed.

---

# 30. Test dei costi

Simulare:

```text
commissioni
spread
slippage
```

e confrontare:

```text
gross return
net return
```

---

# 31. Test della frequenza di rebalance

Confrontare:

```text
1 volta/settimana
2 volte/settimana
3 volte/settimana
daily
```

senza cambiare il modello.

Questo separa la qualità del modello dalla qualità della strategia di utilizzo.

---

# 32. Test prediction → portfolio

Il modello predice; un componente separato decide quanto comprare.

Strategie da confrontare:

### A

Top 5.

### B

Top 10.

### C

Top 20%.

### D

Peso proporzionale alla prediction.

### E

Peso proporzionale a prediction / volatility.

### F

Ranking + volatility scaling.

---

# 33. Limiti di posizione

Valori iniziali da testare:

```text
max single stock = 10%
max sector = 30%
max total positions = 20
min position = 2%
cash >= 0
```

Il cash negativo non deve essere possibile nella simulazione finale.

---

# 34. Leakage test

Creare un test automatico che verifichi:

```text
feature(t) utilizza esclusivamente dati <= t
```

e:

```text
target(t) utilizza esclusivamente dati > t
```

Questo deve diventare parte permanente della pipeline.

---

# 35. Data quality test

Controllare:

- missing values;
- duplicate dates;
- split adjustment;
- stock split;
- dividend adjustment;
- ticker changes;
- delisting;
- timezone;
- trading holidays;
- market closure;
- futures rollover.

---

# 36. Futures: serie continua

Per ogni future deve essere definita chiaramente la politica:

```text
front month
continuous contract
back-adjusted
ratio-adjusted
```

Un cambio di contratto non deve produrre un falso rendimento enorme.

---

# 37. Timing test

Definire precisamente:

```text
t = close del giorno D
```

e quando la previsione diventa disponibile.

L'esecuzione deve essere simulata realisticamente:

```text
close D?
open D+1?
close D+1?
```

Non utilizzare il close di D per simulare un acquisto al close di D se il dato non era disponibile prima della chiusura.

---

# 38. Report obbligatori

## R01 — Dataset Report

```text
numero stock
numero giorni
missing
coverage
settori
```

## R02 — Target Report

Distribuzione di:

```text
Y5
Y10
Y15
```

## R03 — Feature Report

Per ogni feature:

```text
mean
std
min
max
missing
correlation
```

## R04 — Model Report

```text
architecture
parameters
training time
loss
seed
hyperparameters
```

## R05 — Prediction Report

Per ogni giorno:

```text
ticker
prediction_y5
prediction_y10
prediction_y15
actual_y5
actual_y10
actual_y15
```

## R06 — Ranking Report

```text
Spearman
Kendall
Q1...Q5
Q5-Q1
```

## R07 — Portfolio Report

```text
return
Sharpe
Sortino
max drawdown
volatility
turnover
costs
```

## R08 — Sector Report

Performance per settore.

## R09 — Regime Report

Performance per regime di mercato.

## R10 — Experiment Comparison

| Experiment | Spearman | Q5-Q1 | Sharpe | DD | Turnover |
|---|---:|---:|---:|---:|---:|

Questo deve essere il report principale del progetto.

---

# 39. Experiment registry

Ogni esperimento deve avere un ID.

Esempio:

```text
EXP-001
```

Metadata:

```yaml
model: DNN
window: 20
targets: [5, 10, 15]
output_activation: linear
loss: huber
universe_size: 100
features: full
split: walk_forward
seed: 42
```

Il risultato deve essere riproducibile.

---

# 40. Reproducibility

Ogni modello deve poter essere ricreato utilizzando:

```text
experiment_id
dataset_version
code_version
configuration
random_seed
model_version
```

---

# 41. Baseline strategici

Prima di giudicare la rete bisogna battere almeno:

```text
SPY buy & hold
QQQ buy & hold
equal-weight universe
momentum 5d
momentum 20d
random ranking
```

Se la DNN non batte alcune baseline, non c'è motivo di preferirla.

---

# 42. Criterio di successo

Non definire a priori un unico valore come "Sharpe > X".

Utilizzare una gerarchia.

## Livello 1 — Predictive

```text
Spearman > 0
Q5-Q1 > 0
```

e statisticamente significativo.

## Livello 2 — Stable

Il risultato persiste:

- in più periodi;
- con diversi seed;
- in diversi regimi.

## Livello 3 — Economic

Il ranking produce:

```text
positive net return
acceptable drawdown
reasonable turnover
```

## Livello 4 — Competitive

Il sistema supera baseline semplici dopo i costi.

Solo a questo punto il modello è candidato per Hayai.

---

# 43. Statistical significance

Non basta ottenere:

```text
Spearman = +0.04
```

Calcolare:

- confidence interval;
- bootstrap;
- t-statistic dove appropriato;
- distribuzione dei risultati giornalieri;
- eventualmente block bootstrap per rispettare la dipendenza temporale.

Lo stesso vale per Q5-Q1.

---

# 44. Multiple testing

Questo è un rischio molto serio.

Se vengono provati:

```text
50 modelli
×
20 configurazioni
×
10 feature set
```

prima o poi qualcosa sembrerà eccellente per puro caso.

Il test finale deve quindi essere **blind**.

Una parte del dataset deve essere conservata e utilizzata una sola volta alla fine.

---

# 45. Final holdout

Proposta:

```text
TRAIN
VALIDATION
TEST
FINAL HOLDOUT
```

Il `FINAL HOLDOUT` non deve essere utilizzato per scegliere:

- architettura;
- feature;
- hyperparameter;
- universo;
- activation.

Serve esclusivamente alla verifica finale.

---

# 46. Pipeline finale

```text
                    DATA SOURCES
                         │
                         ▼
                DATA QUALITY CHECK
                         │
                         ▼
                UNIVERSE SELECTION
                         │
                         ▼
                 FEATURE BUILDER
                         │
                         ▼
                  20-DAY WINDOWS
                         │
                         ▼
                  TARGET BUILDER
                    Y5/Y10/Y15
                         │
                         ▼
              TRAIN / VALIDATION
                         │
                         ▼
                MODEL TRAINING
                         │
                         ▼
                 WALK-FORWARD
                         │
                         ▼
             PREDICTION DATABASE
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        MODEL METRICS          PORTFOLIO TEST
             │                       │
             └───────────┬───────────┘
                         ▼
                  EXPERIMENT REPORT
                         │
                         ▼
                  MODEL SELECTION
```

---

# 47. Sequenza degli esperimenti iniziali

```text
EXP-001
Baseline momentum

EXP-002
DNN + Y5/Y10/Y15 + Linear

EXP-003
DNN + Y5/Y10/Y15 + Arctan

EXP-004
DNN + solo stock data

EXP-005
DNN + stock + indices

EXP-006
DNN + stock + indices + commodities

EXP-007
DNN + full macro

EXP-008
DNN 10-day window

EXP-009
DNN 20-day window

EXP-010
DNN 40-day window

EXP-011
CNN

EXP-012
GRU

EXP-013
best architecture + relative target

EXP-014
best model + portfolio optimization

EXP-015
final walk-forward

EXP-016
final blind holdout
```

---

# 48. Elementi da definire prima del codice

## A. Definizione esatta dell'universo

Stabilire come selezionare i 60–150 titoli.

Possibili criteri:

```text
market cap
liquidity
average daily volume
sector
exchange
price minimum
```

Non usare semplicemente una lista manuale.

## B. Data source

Decidere la sorgente definitiva dei dati.

Per il prototipo può essere sufficiente Yahoo Finance.

Per una ricerca più seria considerare:

- qualità dello storico;
- corporate actions;
- delisted stocks;
- futures continuous contracts;
- survivorship bias;
- disponibilità storica.

## C. Feature schema definitivo

Definire esattamente:

```text
20 giorni × N features
```

distinguendo:

```text
STOCK FEATURES
MARKET FEATURES
COMMODITY FEATURES
FX FEATURES
RATE FEATURES
```

## D. Portfolio construction

Il modello predice.

Un componente separato decide:

```text
quanto comprare
```

I due problemi devono rimanere separati.

## E. Protocollo di selezione del modello

La scelta finale deve considerare:

```text
1. predictive power
2. ranking power
3. temporal stability
4. robustness
5. portfolio performance
6. costs
7. drawdown
8. simplicity
```

---

# 49. Risultato finale atteso

Alla fine del progetto non vogliamo soltanto:

```text
model.onnx
```

Vogliamo ottenere un modello accompagnato da evidenza sperimentale:

```text
Hayai V2
│
├── dataset specification
├── universe specification
├── feature specification
├── target specification
├── experiment registry
├── trained models
├── predictions
├── statistical tests
├── ranking analysis
├── portfolio simulations
├── robustness tests
├── final holdout
└── model selection report
```

Il prodotto finale deve poter rispondere quantitativamente a:

> **Perché dovrei fidarmi di questo modello più che di un semplice momentum model?**

Se non possiamo rispondere a questa domanda con i dati degli esperimenti, il progetto non è ancora concluso.

---

# 50. Specifica baseline ufficiale

```text
UNIVERSE
60–150 stocks
target ≈ 100

INPUT
20 trading days

DATA
Stock OHLCV
+
indices
+
oil
+
copper
+
gold
+
silver
+
rates
+
FX

TARGET
Y5
Y10
Y15

MODEL
Dense 128 ReLU
Dense 64 ReLU
Dense 32 ReLU
Output 3

EXPERIMENT
Linear vs Scaled Arctan

VALIDATION
Walk-forward

PRIMARY METRICS
Daily Spearman
Q5-Q1

SECONDARY METRICS
MAE
RMSE
Hit Rate
Sharpe
Max Drawdown
Turnover

PORTFOLIO
Stocks only
```

---

# 51. Decisioni metodologiche ancora aperte

Le seguenti decisioni non devono essere fissate arbitrariamente prima degli esperimenti:

1. numero ottimale di stock tra 60 e 150;
2. 10/20/40/60 giorni di finestra;
3. target assoluto vs relativo;
4. Linear vs Scaled Arctan;
5. MSE vs Huber;
6. DNN vs CNN vs GRU vs LSTM;
7. numero e tipologia di variabili macro;
8. schema di portfolio construction;
9. frequenza ottimale di rebalance;
10. eventuale utilizzo di short;
11. limiti per singolo titolo e settore.

Queste sono **variabili sperimentali**, non decisioni da prendere in base a intuizioni.

---

# 52. Principio conclusivo

Il progetto deve seguire questa regola:

> **Prima misurare, poi decidere.**

L'obiettivo non è trovare il modello più complesso.

L'obiettivo è trovare il modello più semplice che dimostri, su dati realmente out-of-sample e dopo costi realistici, una capacità di ranking:

```text
prediction ↑
      ↓
future return ↑
```

stabile nel tempo, nei settori e nei diversi regimi di mercato.

Solo un risultato di questo tipo giustifica l'integrazione del modello nel sistema Hayai.
