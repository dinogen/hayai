# 10 — Glossario

Glossario dei termini chiave usati nell'applicazione e nella documentazione.

## A

- **actual position** — posizione effettivamente detenuta (o cartacea, nel caso
  yfinance). File `f008_actual.parquet` o, su Alpaca, `get_all_positions()`.
- **asset** — sinonimo di simbolo/titolo (azioni) nel portafoglio.

## B

- **batch** — esecuzione a lotti, non interattiva: ogni esecuzione della CLI esegue
  una o più fasi della pipeline.
- **build signals** — fase CLI (`-s`) che applica il modello e definisce i pesi.
- **buy** — ordine di acquisto.

## C

- **cash flow** — flusso di cassa generato da un ordine: `-qty_diff * price`.
  Negativo per acquisti, positivo per vendite.
- **CASH_SYMBOL / MYCASH** — simbolo speciale (`MYCASH`) che rappresenta la
  liquidità del portafoglio; nei parquet ha `qty=1` e `price` = valore della cassa.
- **clip** — troncare i valori a un intervallo (`clip_min`, `clip_max`).
- **close** — ordine di chiusura totale di una posizione.
- **conf.ini** — file di configurazione (portfolio o modello), letto da
  `create_context`.
- **context** — dizionario globale creato da `hayai_util.create_context` con tutte
  le configurazioni; `util.context` è usato ovunque.

## D

- **data_source** — parametro che sceglie il provider dati: `yfinance` o `alpaca`.
- **define_orders** — funzione che genera il file ordini `f007_orders.parquet`.
- **define_weight** — funzione che calcola i pesi target `f003_weights.parquet`.
- **dummies (one-hot)** — variabili binarie per Paese (`country_*`) e Settore
  (`sector_*`).
- **dry-run** — esecuzione di prova senza effetti (concetto da introdurre nel nuovo
  sistema).

## E

- **equity** — valore totale del portafoglio (cassa + posizioni).
- **execution** — fase CLI (`-e`) che invia gli ordini al broker.

## F

- **f0NN_*.parquet** — prefisso numerico dei file intermedi (`f001_features`,
  `f002_predictions`, `f003_weights`, `f005_position_new`,
  `f006_position_new_qty`, `f007_orders`, `f008_actual`). `f004` è definito ma non
  usato.
- **feature** — variabile di input del modello (derivata dai prezzi/volumi e da
  fattori esterni).
- **forex** — coppie di valute scaricate da yfinance (configurate nel modello).
- **FTSEMIB.MI** — simbolo yfinance dell'indice FTSE MIB.

## G

- **get_latest_price** — recupera l'ultimo prezzo disponibile per un insieme di
  simboli (yfinance o Alpaca).
- **get_equity** — recupera l'equity (reale su Alpaca, stimata/rotta su yfinance).

## H

- **hist** — cartella con uno storico parquet per simbolo (`hist/{SYMBOL}.parquet`).
- **hl_range** — feature `(high - low) / close`.

## I

- **indexes** — indici scaricati da yfinance (configurati nel modello).
- **ingestion** — fase CLI (`-i`) che scarica quote e calcola le feature.
- **init_portfolio** — inizializza `f008_actual.parquet` con solo cassa.

## L

- **label_min / label_max** — estremi del target usati per la denormalizzazione
  delle predizioni (salvati in conf.ini dal training).
- **log_return** — `ln(close_t / close_{t-k})`; base del target.
- **long** — posizione lunga (acquisto, `qty > 0`).

## M

- **model portfolio** — portfolio speciale (id `model_*`) usato per addestrare un
  modello; contiene `model.keras`, `mins.csv`, `maxs.csv`, `conf.ini`,
  `portfolio.csv`.
- **mom (momentum)** — variazione percentuale su una finestra: `mom_5/10/20`.
- **MYCASH** — vedi CASH_SYMBOL.

## N

- **n_long / n_short** — numero di asset long/short nei pesi.
- **notional** — importo monetario usato da `init_portfolio.py` per comprare quote
  uguali di ogni simbolo.

## O

- **order** — istruzione di trading (BUY/SELL/CLOSE) con quantità e prezzo.
- **outlier** — valore anomalo; viene clippato ai quantili 1% e 99%.

## P

- **paper trading** — trading su account di simulazione Alpaca (`paper=True`).
- **parquet** — formato di file colonnare usato per tutti i dati intermedi.
- **portfolio** — insieme di simboli con configurazione propria; ogni portfolio ha
  una cartella in `data/`.
- **prediction** — output del modello denormalizzato e clippato.
- **price** — prezzo corrente usato per calcolare quantità e ordini.

## Q

- **qty_diff_perc_min** — soglia minima (default 0.2) sulla variazione percentuale
  sotto la quale non si genera un ordine.
- **qty_new / qty_old / qty_diff** — quantità nuova, vecchia, differenza.

## R

- **rebalance** — ribilanciamento: portare le quantità reali verso quelle target.
- **risk_percentage** — quota di equity investita (default 0.8).
- **report** — fase CLI (`-r`); genera `report_{portfolio_id}.html`.

## S

- **secret.ini** — file con credenziali (Telegram globale, Alpaca per portfolio).
- **sell** — ordine di vendita.
- **short** — posizione corta (vendita allo scoperto, `qty < 0`).

## T

- **target** — variabile da predire: `clip(ln(close_{t+trd}/close_t)/vol_20)`.
- **target_return_days (trd)** — orizzonte del target (default 5).
- **time_in_force (DAY)** — validità dell'ordine Alpaca (giorno corrente).
- **trading portfolio** — portfolio di trading vero e proprio (sottoinsieme del
  model portfolio).

## V

- **vol (volatility)** — volatilità: `vol_10`, `vol_20` (std del log_return).
- **vol_regime** — `vol_10 / vol_60`.
- **volume_shock** — `volume / MA20(volume)`.

## W

- **weight** — peso target di un asset: `prediction.clip / vol_20`, normalizzato a
  somma |pesi| = 1.
- **webapp** — applicazione web da costruire nel nuovo progetto (dashboard, config,
  approvazione ordini).

## Y

- **yfinance** — libreria Python per i dati Yahoo Finance.
