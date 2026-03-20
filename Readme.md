# HAYAI trading model and execution

## Costruzione portfolio

Si prende da https://www.nasdaq.com/market-activity/stocks/screener.
Per il portafoglio del modello si filtra in modo da avere 3-4 country e 3-4 sectors in modo da arrivare a  circa 1000 symbol.
Per i portafogli si prende un sottoinsieme del portafoglio del modello
In questo modo non ci sono i problemi legati al Country e al Sector.

## Portafogli

1. Usa e Canada; Tech, Telecom, Financial e Energy
2. Usa e Canada; Health Utilities  Consumer

# File della posizione
Nome file: actual_position.parquet
Viene calcolato a mercato aperto in base al prezzo flash.
Contiene la posizione per ogni data.
Il symbol speciale MYCASH viene diminuito ad ogni acquisto e viene aumentato ad ogni vendita.

## Colonne
data, symbol, qty, price,        value

- data: data dell'execution
- price: prezzo di acquisto o di vendita
- value = qty * price

C'è una riga speciale:
data, MYCASH,   1,   cash_residuo, cash_residuo


# ISSUES

1. OK separare i conf.ini. Mettere le chiavi in dei file secret.ini
1. togliere la colonna volume dal set delle features
1. Per scaricare i dati serve solo symbol, close, volume. 
1. creare più modelli, non uno solo, e ogni portafoglio indica il modello che usa
1. funzione che calcola la posizione e la equity dai dati scaricati.
1. OK creazione di un report con la posizione calcolata in loco
1. scaricare i dati da yfinance o da alpaca a seconda di un parametro per prendere anche i dati Europei



