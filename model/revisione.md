Ti preparo un documento operativo pensato per produrre un **log diagnostico standardizzato** che poi possiamo analizzare insieme. L'obiettivo è separare chiaramente problemi del modello, della costruzione del portafoglio e dell'esecuzione.

# Hayai — Piano di test diagnostico del modello

## Obiettivo

L'obiettivo di questa analisi è capire perché il paper portfolio di Hayai, partendo da €5.000, è sceso intorno a €4.200 e non riesce a recuperare.

Non modificare il modello prima di aver completato questi test.

La prima fase deve essere **diagnostica**, non di ottimizzazione.

Il risultato finale deve essere un unico report Markdown contenente dati e risultati dei test, da poter analizzare successivamente.

---

# 1. Informazioni generali dell'esperimento

Registrare:

* data/ora dell'esperimento
* versione del codice Git / commit
* versione del modello
* configurazione utilizzata
* periodo coperto dal dataset
* periodo di training
* periodo di validation
* periodo di test
* numero di asset
* capitale iniziale
* frequenza di rebalance
* commissioni
* eventuale slippage
* eventuale short selling
* eventuale leverage

Esempio:

```text
Experiment date: 2026-09-09
Git commit: abc123
Model: stock_model/v2
Universe: 500 assets
Initial capital: €5000
Rebalance: weekly
Transaction cost: ...
Slippage: ...
```

---

# 2. Test di integrità del dataset

Verificare:

* numero di righe
* numero di asset
* date min/max
* valori mancanti
* duplicati
* prezzi <= 0
* feature infinite
* feature NaN
* target NaN
* eventuali asset con serie temporali troppo corte

Riportare:

```text
Rows:
Assets:
First date:
Last date:

Missing values:
Duplicate rows:
Invalid prices:
Infinite values:
NaN features:
NaN targets:
```

---

# 3. Verifica del target

Il target attuale è basato sul rendimento futuro a 5 giorni.

Verificare esplicitamente:

```text
target(t) = return(t → t+5)
```

Controllare:

* distribuzione del target
* media
* mediana
* deviazione standard
* min
* max
* percentili 5/25/50/75/95
* percentuale target > 0
* percentuale target < 0

Riportare inoltre alcuni esempi manuali:

```text
Asset: XXX
Date: YYYY-MM-DD
Close(t):
Close(t+5):
Calculated target:
Stored target:
```

Questo serve a verificare che non ci siano errori nel calcolo del target.

---

# 4. Verifica temporale delle feature

Questo è un test fondamentale.

Per ogni feature verificare che al tempo `t` utilizzi esclusivamente informazioni disponibili a `t`.

In particolare controllare:

* rolling windows
* momentum
* volatilità
* beta
* correlation
* market features
* SPY features
* breadth
* dispersion
* ranking cross-sectional

Per ogni feature sospetta documentare:

```text
Feature:
Input data:
Lookback:
Uses future information: YES/NO
```

Se una feature utilizza anche solo indirettamente dati futuri, segnalarla come:

```text
LOOK-AHEAD BIAS
```

---

# 5. Verifica dello split train/validation/test

Questo test è prioritario.

Determinare esattamente come vengono separati i dati.

Il modello finanziario deve essere valutato temporalmente.

Riportare:

```text
Training:
    start:
    end:
    rows:

Validation:
    start:
    end:
    rows:

Test:
    start:
    end:
    rows:
```

Verificare inoltre:

```text
Random split: YES/NO
Chronological split: YES/NO
Overlap between periods: YES/NO
```

Se viene utilizzato uno split casuale, segnalarlo chiaramente.

---

# 6. Test baseline

Prima di valutare il modello, calcolare alcune strategie molto semplici.

### Baseline 1 — Buy & Hold

Calcolare il rendimento di:

* SPY
* eventuale benchmark principale
* eventuale universo equally weighted

### Baseline 2 — Random

Generare segnali casuali con la stessa frequenza del modello.

### Baseline 3 — Momentum semplice

Utilizzare soltanto una feature semplice, ad esempio:

```text
20-day momentum
```

### Baseline 4 — Zero model

Allocazione neutrale / cash.

Riportare:

| Strategy     | Return | Volatility | Max Drawdown |
| ------------ | -----: | ---------: | -----------: |
| Model        |        |            |              |
| SPY          |        |            |              |
| Equal Weight |        |            |              |
| Random       |        |            |              |
| Momentum     |        |            |              |

Il modello deve essere confrontato con qualcosa di semplice.

---

# 7. Test della capacità predittiva del modello

Questo è probabilmente il test più importante.

Non utilizzare ancora il portfolio.

Per ogni previsione salvare:

```text
date
symbol
prediction
actual_return_1d
actual_return_5d
actual_return_10d
```

Calcolare la correlazione tra:

```text
prediction
```

e:

```text
future_return_5d
```

Calcolare:

* Pearson correlation
* Spearman correlation
* MAE
* RMSE

Riportare:

```text
Pearson:
Spearman:
MAE:
RMSE:
```

---

# 8. Test dei quintili del signal

Per ogni data:

1. ordinare tutti gli asset in base al signal;
2. dividere gli asset in 5 gruppi;
3. calcolare il rendimento futuro medio di ogni gruppo.

Esempio:

| Quintile     | Signal | Future 5d return |
| ------------ | -----: | ---------------: |
| Q1 — lowest  |        |                  |
| Q2           |        |                  |
| Q3           |        |                  |
| Q4           |        |                  |
| Q5 — highest |        |                  |

Calcolare anche:

```text
Q5 - Q1
```

Questo test deve essere fatto **senza portfolio construction**.

L'obiettivo è capire se:

```text
signal alto → rendimento futuro alto
```

---

# 9. Test Long vs Short

Separare i segnali positivi e negativi.

Calcolare:

```text
LONG:
mean future return
median future return
win rate

SHORT:
mean future return
median future return
win rate
```

Per lo short considerare il rendimento dal punto di vista della posizione short.

Riportare:

```text
Long edge:
Short edge:
```

Questo permette di capire se il modello funziona soltanto in una direzione.

---

# 10. Test del signal per intervalli

Dividere i segnali in intervalli:

```text
[-3,-2]
[-2,-1]
[-1,0]
[0,1]
[1,2]
[2,3]
```

Per ogni intervallo calcolare:

* numero di osservazioni
* signal medio
* rendimento futuro medio
* win rate

Esempio:

| Signal range |  N | Future 5d return | Win rate |
| ------------ | -: | ---------------: | -------: |
| -3/-2        |    |                  |          |
| -2/-1        |    |                  |          |
| -1/0         |    |                  |          |
| 0/1          |    |                  |          |
| 1/2          |    |                  |          |
| 2/3          |    |                  |          |

Questo permette di capire se il modello ha una relazione monotona con il rendimento.

---

# 11. Test del modello originale vs news correction

Questo test deve separare chiaramente la componente quantitativa dalla componente LLM.

Per ogni previsione registrare:

```text
quant_signal
news_modifier
final_signal
```

Calcolare separatamente le performance di:

```text
quant_signal
quant_signal + news_modifier
```

Riportare:

| Model        | Return | Sharpe | Max DD | Win rate |
| ------------ | -----: | -----: | -----: | -------: |
| Quant        |        |        |        |          |
| Quant + News |        |        |        |          |

Inoltre:

```text
Mean absolute news modifier:
Max news modifier:
% observations modified:
```

---

# 12. Test della news correction

Analizzare la news correction indipendentemente.

Dividere le operazioni in:

```text
News strongly positive
News mildly positive
No significant news
News mildly negative
News strongly negative
```

Per ciascun gruppo calcolare:

```text
number of observations
average modifier
future 5d return
win rate
```

Obiettivo:

verificare se il modifier prodotto dall'LLM ha effettivamente una relazione con il rendimento futuro.

---

# 13. Test signal → portfolio weight

Questo test serve a capire se il problema viene introdotto dal portfolio construction.

Per ogni asset registrare:

```text
signal
target_weight
```

Calcolare:

```text
correlation(signal, target_weight)
```

e verificare:

```text
signal alto → weight alto
signal basso → weight basso
```

Riportare inoltre:

```text
Maximum weight:
Minimum weight:
Mean absolute weight:
Number of positions:
Long exposure:
Short exposure:
Gross exposure:
Net exposure:
```

---

# 14. Test della concentrazione

Per ogni rebalance calcolare:

```text
largest position
top 3 positions
top 5 positions
Herfindahl index
```

Riportare la media e il massimo.

Particolare attenzione alla posizione BND.

Determinare:

```text
Average BND allocation:
Maximum BND allocation:
Minimum BND allocation:
```

e verificare perché BND riceve quella quantità di capitale.

---

# 15. Test del turnover

Calcolare per ogni rebalance:

```text
turnover = sum(abs(target_weight - current_weight))
```

Riportare:

```text
Average turnover:
Median turnover:
Maximum turnover:
```

Stimare anche quanto del rendimento viene perso a causa di:

* commissioni
* spread
* slippage

---

# 16. Test del timing

Per ogni trade registrare:

```text
signal timestamp
signal price
execution timestamp
execution price
next-day price
5-day price
```

Verificare esplicitamente:

```text
Signal generated before execution: YES/NO
Execution uses future price: YES/NO
Target return window matches trading window: YES/NO
```

Questo test è fondamentale perché il modello prevede un rendimento futuro di 5 giorni.

Bisogna verificare che il trade effettivamente effettuato corrisponda a quella previsione.

---

# 17. Test del paper portfolio

Ricostruire il portafoglio **da zero** partendo da:

```text
Initial capital = €5000
```

utilizzando esclusivamente:

```text
signals
target weights
execution prices
fees
slippage
```

Confrontare il NAV ricostruito con il NAV reale del sistema.

Riportare:

```text
System NAV:
Reconstructed NAV:
Difference:
```

Se la differenza è significativa, il problema potrebbe essere nell'implementazione del portfolio/accounting e non nel modello.

---

# 18. Analisi trade-by-trade

Produrre una tabella con almeno:

| Date | Symbol | Quant | News | Final | Side | Entry | Exit | Return |
| ---- | ------ | ----: | ---: | ----: | ---- | ----: | ---: | -----: |

Calcolare:

```text
Number of trades:
Winning trades:
Losing trades:
Win rate:
Average winner:
Average loser:
Largest winner:
Largest loser:
Profit factor:
```

---

# 19. Analisi per asset

Per ogni asset:

```text
number of trades
total P&L
average P&L
win rate
long P&L
short P&L
```

Produrre una classifica:

```text
BEST ASSETS
1.
2.
3.
...

WORST ASSETS
1.
2.
3.
...
```

---

# 20. Analisi temporale

Calcolare rendimento:

* giornaliero
* settimanale
* mensile

Riportare:

```text
Best month:
Worst month:
Best week:
Worst week:
Longest losing streak:
Longest winning streak:
```

Calcolare inoltre:

```text
Cumulative return
Maximum drawdown
Recovery time
```

---

# 21. Test di stabilità temporale

Dividere il periodo di test in sotto-periodi.

Ad esempio:

```text
Period 1
Period 2
Period 3
Period 4
```

Per ciascuno calcolare:

```text
return
Sharpe
win rate
Q5-Q1 spread
```

L'obiettivo è capire se il modello funziona:

```text
sempre
```

oppure soltanto:

```text
in alcuni periodi di mercato
```

---

# 22. Test per regime di mercato

Se possibile classificare il mercato come:

```text
Bull
Bear
Sideways
High volatility
Low volatility
```

e calcolare la performance del modello per ciascun regime.

Esempio:

| Regime   | Return | Sharpe | Win rate |
| -------- | -----: | -----: | -------: |
| Bull     |        |        |          |
| Bear     |        |        |          |
| Sideways |        |        |          |
| High vol |        |        |          |
| Low vol  |        |        |          |

---

# 23. Test di robustezza

Senza modificare il modello principale, verificare cosa succede cambiando:

### Holding period

```text
1 day
5 days
10 days
20 days
```

### Number of positions

```text
5
10
20
50
```

### Signal threshold

```text
0
0.25
0.5
1.0
```

### News modifier

```text
disabled
0.05
0.10
0.20
```

L'obiettivo non è trovare il parametro migliore.

L'obiettivo è capire se il risultato è **robusto**.

---

# 24. Test di inversione del signal

Questo test è molto importante.

Calcolare la performance di:

```text
original signal
```

e:

```text
-inverted signal
```

Se il signal originale perde sistematicamente e quello invertito guadagna, potrebbe esserci un problema nella definizione del target, nella direzione del signal o nella trasformazione prediction → position.

Riportare:

```text
Original:
Inverted:
```

---

# 25. Test del modello contro un modello banale

Confrontare il modello con:

```text
Momentum 20d
Momentum 60d
Momentum 120d
```

Se una semplice strategia momentum batte nettamente la rete neurale, questo è un risultato importante.

Non è necessariamente un fallimento: significa che la complessità del modello non sta producendo valore aggiuntivo.

---

# 26. Metriche finali da riportare

Il report finale deve contenere almeno:

```text
Initial capital:
Final NAV:
Total return:
Annualized return:
Volatility:
Sharpe:
Maximum drawdown:
Win rate:
Profit factor:
Average trade:
Number of trades:
Turnover:
Transaction costs:
```

---

# 27. Giudizio diagnostico automatico

Alla fine del report generare una sezione:

```markdown
## Diagnostic conclusion

### Model predictive power
GOOD / WEAK / NONE / INVERTED

### Data leakage
YES / NO / UNCERTAIN

### Portfolio construction
GOOD / SUSPICIOUS / BAD

### Execution
GOOD / SUSPICIOUS / BAD

### News correction
POSITIVE / NEUTRAL / NEGATIVE / UNCERTAIN

### Main suspected problem
...

### Evidence
1. ...
2. ...
3. ...

### Recommended next investigation
...
```

Questa sezione non deve cercare di "aggiustare" il modello.

Deve soltanto riassumere ciò che i test hanno evidenziato.

---

# 28. Log delle singole predizioni

Oltre al report aggregato, creare un CSV con una riga per ogni prediction.

Formato minimo:

```text
timestamp
date
symbol
quant_signal
news_modifier
final_signal
target_weight
side
price_at_signal
price_at_execution
return_1d
return_5d
return_10d
```

Questo file è particolarmente importante.

Se possibile, aggiungere anche:

```text
news_score
news_confidence
news_age
model_version
```

---

# 29. Regola fondamentale

Non eliminare i risultati negativi.

Non modificare retroattivamente:

* segnali
* target
* prezzi
* news score
* portfolio weights

Se un test produce un risultato negativo, deve rimanere nel report.

Il report deve rappresentare esattamente ciò che Hayai avrebbe saputo **in quel momento**.

---

# 30. Output richiesto

Produrre questi due file:

```text
hayai_diagnostic_report.md
hayai_predictions.csv
```

Il file Markdown deve contenere tutti i test precedenti.

Il CSV deve contenere le singole predizioni.

Quando il test è completato, fornire entrambi i file.

Il `hayai_diagnostic_report.md` sarà il documento principale da analizzare.

---

# Priorità

Se eseguire tutti i test è troppo oneroso, eseguire prima questi:

1. **Train/validation/test split**
2. **Look-ahead bias**
3. **Prediction vs actual 5-day return**
4. **Quintile analysis**
5. **Long vs Short**
6. **Quant vs Quant + News**
7. **Signal → Portfolio Weight**
8. **Portfolio reconstruction**
9. **Trade-by-trade P&L**
10. **Original vs inverted signal**

Questi dieci test dovrebbero essere sufficienti per determinare se il problema principale è:

```text
DATA
  ↓
FEATURES
  ↓
MODEL
  ↓
SIGNAL
  ↓
NEWS
  ↓
PORTFOLIO
  ↓
EXECUTION
  ↓
P&L
```

e individuare il punto in cui nasce la perdita.
