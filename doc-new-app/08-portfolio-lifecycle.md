# 08 — Ciclo di Vita del Portafoglio (Bootstrapping, Simulazione e Valutazione Giornaliera)

Questo documento risponde a una domanda fondamentale: **Come evolve il portafoglio giorno per giorno, partendo da zero fino alla gestione dinamica degli asset?**

Dato che l'applicazione gestisce un esperimento da **€5.000** con un approccio *human-in-the-loop*, il sistema deve simulare il valore e la composizione del portafoglio giorno dopo giorno in base ai segnali generati.

---

## 1. La Fase di Impianto (Bootstrap / Giorno 1)

Prima di far girare i modelli, dobbiamo definire l'**Universo di Partenza (Watchlist)** e inizializzare lo stato finanziario.

### 1.1 Come si sceglie la lista iniziale di Azioni, ETF e Bond?
All'inizio ci creiamo una lista curata di base (es. 25-30 strumenti altamente liquidi e rappresentativi):
- **Azioni Large Cap**: es. AAPL, MSFT, NVDA, GOOGL, ASML, ENEL (titoli con elevata liquidità e notizie frequenti).
- **ETF Settoriali/Geografici**: es. QQQ (Nasdaq), SPY (S&P 500), VGK (Europa), EEM (Emergenti).
- **ETF Obbligazionari & Bond Yields**: es. BND (Aggregate Bond), TLT (Treasury Long-Term), e il rendimento del Treasury a 10 anni (`^TNX`).

### 1.2 Universo di Training vs Watchlist (chiarimento fondamentale)

È importante distinguere tre concetti che spesso vengono confusi:

1. **Universo di Training (~100 asset)**: un pool ampio di dati (es. 100 strumenti,
   4-5 anni di storico) usato **solo per addestrare il modello Keras**. Non è il
   portafoglio e non implica che tutti questi titoli vengano detenuti.
2. **Watchlist (25-30 strumenti)**: l'universo investibile di partenza, associato
   al portafoglio. È la lista su cui ogni notte vengono calcolati i segnali.
3. **Posizioni detenute**: solo gli strumenti selezionati (top `n_long` / bottom
   `n_short`) ricevono un peso > 0. Tutti gli altri hanno **peso 0** e restano in
   watchlist come candidati per i rebalance futuri.

### 1.3 Stato al Giorno 1 (Bootstrap)
- **Liquidità (Cash)**: €5.000,00 (100%).
- **Posizioni detenute**: Zero quote per tutti gli asset.
- **Database**: Vengono inseriti gli strumenti in `instrument` e associati al portafoglio in `portfolio_instrument`. Il batch `data` scarica lo storico degli ultimi 5 anni per permettere al modello di calcolare le feature e le volatilità.

---

## 2. L'Evoluzione Dinamica (Giorno 2, Giorno 3, ...)

Il sistema non è statico: ogni giorno la composizione target può variare in base alle predizioni di Keras e al sentiment di DeepSeek.

### 2.1 Il Ciclo Giornaliero di Valutazione (Mark-to-Market)
Ogni notte, il batch esegue questi passaggi finanziari:
1. **Aggiornamento Prezzi**: Scarica la chiusura del giorno per tutti gli strumenti dell'universo (`price_daily`).
2. **Metadati Asset** (job `metadata`, con cadenza meno frequente es. settimanale): aggiorna `sector`, `country` e `area` degli strumenti dalla watchlist via yfinance (solo se mancanti o più vecchi di 30 giorni). L'`area` deriva dalla `country` con priorità Emergenti > EU > USA > Asia > Altro, con fallback manuale per simbolo per ETF/bond yield.
3. **Calcolo Segnali**: Esegue Keras + DeepSeek → `portfolio_signal`.
4. **Ottimizzazione Pesi**: Seleziona i top `n_long` e bottom `n_short`, normalizzando i pesi a 1.0 (`portfolio_recommendation`).
5. **Valutazione del Portafoglio (NAV — solo mark-to-market)**:
   - Se il portafoglio ha posizioni aperte, il sistema calcola il valore di mercato attuale di ciascuna quota (`qty × prezzo odierno`) **senza modificare quantità e costo di carico**: le posizioni del portafoglio attuale sono gestite manualmente dalla pagina "Portafoglio Attuale".
   - Somma la liquidità disponibile (`portfolio_cash`, aggiornata solo dalle operazioni manuali).
   - Determina il **Valore Totale del Portafoglio (NAV)** aggiornato a quella data.

---

## 3. Tracciamento di Posizioni e Valore in MariaDB

Per permettere alla webapp di mostrare ogni giorno la composizione e il valore del portafoglio, introduciamo due concetti chiave nel database:

1. **`portfolio_cash`**: Saldo della liquidità disponibile.
2. **`portfolio_position`**: Le quote effettivamente "detenute" nel portafoglio simulato (o sincronizzate con le decisioni prese con il promotore).

Quando il martedì decidi di seguire le raccomandazioni del sistema e compri/vendi con il promotore, registri l'operazione nella pagina **"Portafoglio Attuale"** (apertura/chiusura/modifica posizioni) oppure la esegui **riga per riga** dalla **Tabella di Riconciliazione** della pagina "Composizione Consigliata" (bottone **"Esegui"** → `POST /holdings/execute`, che rilegge l'ultima raccomandazione dal DB). Ogni operazione viene salvata in `portfolio_trade` e il cash ricalcolato; da quel momento in poi, la webapp calcola automaticamente il **Mark-to-Market giornaliero** senza alterare le posizioni manuali.

> **Allineamento automatico settimanale**: il job batch `align` (martedì alle 15:20, vedi
> `07-operativita-batch.md`) esegue automaticamente questo allineamento alle raccomandazioni,
> rispettando la soglia di tolleranza `rebalance_threshold_eur` e la guardia anti-stale. Resta
> comunque possibile l'allineamento manuale dalla pagina "Portafoglio Attuale".

> **Regola short interi**: le posizioni short sono sempre espresse in **quote intere**
> (arrotondamento aritmetico half-up, `floor(x + 0.5)`), sia nelle raccomandazioni sia
> nel portafoglio attuale. Se la quantità short arrotondata è **0**, la posizione si
> considera chiusa (la raccomandazione non viene emessa / la posizione viene azzerata).

---

## 4. Aggiornamento Dinamico dell'Universo (Aggiunta/Rimozione Asset)

Nel corso dei mesi, potresti voler aggiungere un nuovo ETF o rimuovere un'azione che non ti interessa più. La gestione avviene **dalla pagina Watchlist** della webapp:

- **Aggiunta**: scegli uno strumento dall'**universo dei candidati** (endpoint `GET /api/portfolios/{code}/universe`, popolato dal job batch `universe` con i ~100 simboli di training, **non** linkati al portafoglio) e premi **"+ Aggiungi"**: viene creato il link in `portfolio_instrument` (`POST /api/portfolios/{code}/watchlist`). Dalla notte successiva, il batch scaricherà lo storico e inizierà a valutarlo insieme agli altri. Se il simbolo non è ancora in universo, lo si può aggiungere digitandolo nella pagina Watchlist (`POST /api/portfolios/{code}/universe`): viene inserito in `instrument` (active=1) da yfinance e diventa subito selezionabile come candidato.
- **Rimozione**: premi **"Rimuovi"** sulla riga (`DELETE /api/portfolios/{code}/watchlist/{instrument_id}`): lo strumento viene **scollegato** dal portafoglio (unlink) e smette di essere processato dal batch notturno, ma **resta nell'universo** (`instrument.active = 1`) e può essere riaggiunto in futuro. Se lo strumento ha una **posizione aperta** (`portfolio_position` con `qty != 0`) la rimozione è **bloccata** (bottone disabilitato nell'UI e 422 dal backend): devi prima chiudere la posizione nella pagina "Portafoglio Attuale".
