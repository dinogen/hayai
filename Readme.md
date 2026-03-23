# HAYAI trading model and execution

## Costruzione portfolio

Si prende da https://www.nasdaq.com/market-activity/stocks/screener.
Per il portafoglio del modello si filtra in modo da avere 3-4 country e 3-4 sectors in modo da arrivare a  circa 1000 symbol.
Per i portafogli si prende un sottoinsieme del portafoglio del modello
In questo modo non ci sono i problemi legati al Country e al Sector.

## Portafogli

1. Usa e Canada; Tech, Telecom, Financial e Energy
2. Usa e Canada; Health Utilities  Consumer

# Files

## portfolio.csv
Colonne: symbol, country, sector
E' scaricato da nasdaq.

## hist/XXXX.parquet
Colonne: date, symbol, close, volume
Un file per ogni symbol XXXX.

## features.parquet
Colonne: date, symbol, [features], target
Una riga per symbol per data


## predictions.parquet
Colonne: come features - target + prediction
Una riga per symbol solo per l'ultima data valida.

## weights.parquet
Colonne: symbol,prediction,vol_20,weight
Una riga per ogni symbol di portafoglio, molti a 0.
I pesi devono fare somma 1.

## position_new.parquet
Colonne: symbol, weight_new, qty_old
Una riga per ogni symbol del portafoglio.

## position_new_qty.parquet
Colonne: symbol, weight_new, qty_old, price, value_new, qty_new, qty_diff, qty_diff_perc
Dal peso e prezzo si calcola qty_new, poi la diff e la percentuale.
Una riga per ogni symbol del portafoglio.


## orders.parquet
Colonne: date, n, symbol, qty, price
Dal file position_new_qty si calcola l'ordine da fare.
Se si cambia segno, si chiude la posizione e si riapre.
Se non si cambia segno si compra o si vende.
Se si atterra a 0 si chiude.
Se si viene da 0 si apre.
orders file deve contenere la storia delle compere e delle vendite.
qty è positivo per buy e negativo per sell
il price è preso flash.
la colonna n serve a ordinare la sequenza degli ordini che è influente.

## actual_position.parquet
Colonne: date, symbol, qty, price, value
Il file actual_position contiene la posizione attuale per data ed è calcolato in base a orders.
Il capitale iniziale sta nel file di conf.ini e nella prima riga CASH a t=0 degli orders
- price: prezzo di acquisto o di vendita
- value = qty * price



# ISSUES

1. OK separare i conf.ini. Mettere le chiavi in dei file secret.ini
1. togliere la colonna volume dal set delle features
1. Per scaricare i dati serve solo symbol, close, volume. 
1. creare più modelli, non uno solo, e ogni portafoglio indica il modello che usa
1. funzione che calcola la posizione e la equity dai dati scaricati.
1. OK creazione di un report con la posizione calcolata in loco
1. scaricare i dati da yfinance o da alpaca a seconda di un parametro per prendere anche i dati Europei



