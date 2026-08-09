# 06 — API (FastAPI) & Webapp (Angular)

Questo documento definisce l'interfaccia **FastAPI** (backend REST in sola lettura) e
la struttura della **webapp Angular**, focalizzandosi sulla visualizzazione delle
predizioni, delle raccomandazioni e soprattutto delle **schede tesi di investimento**
per la revisione del martedì.

---

## 1. API REST (FastAPI)

Il backend è un'applicazione FastAPI leggera, in sola lettura, collegata a MariaDB.
Gira su uvicorn (`127.0.0.1:8000`) ed è esposta al browser tramite `nginx`.

### Endpoints Principali

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/api/health` | GET | Stato del servizio e data dell'ultimo job in `job_run` |
| `/api/portfolios` | GET | Elenco dei portafogli attivi |
| `/api/portfolios/{code}` | GET | Dettaglio del portafoglio (parametri, strumenti associati). Ogni strumento include anche `sector`, `country`, `area` e `metadata_date` (metadati dal job `metadata`). Filtro opzionale `?area=eu|usa|asia|emerging|other` per limitare gli strumenti a una specifica area geografica |
| `/api/portfolios/{code}/recommendations/latest` | GET | **Composizione consigliata (ultima data)**: pesi, side, importi, variazioni vs settimana precedente |
| `/api/portfolios/{code}/holdings` | GET | **Portafoglio attuale**: posizioni detenute (long/short), P&L, cash, NAV, watchlist e ultime raccomandazioni |
| `/api/portfolios/{code}/holdings/save` | POST | **Salvataggio portafoglio attuale**: applica lo stato desiderato (diff → `portfolio_trade` → snapshot posizioni → ricalcolo cash) |
| `/api/portfolios/{code}/signals` | GET | Segnali ibridi (Quant Score + Sentiment IA) per strumento |
| `/api/portfolios/{code}/news` | GET | Notizie recenti collegate agli strumenti del portafoglio. Parametri opzionali: `?days=14` (retention, default 14), `?sector=` (filtro settore), `?symbol=` (filtro ticker), `?limit=50` (limite righe). Ogni notizia include `sector`, `area`, `sentiment`, `confidence` |
| `/api/news/{news_id}` | GET | Dettaglio di una singola notizia (titolo, publisher, data, summary, link originale, sentiment e rationale IA) |
| `/api/portfolios/{code}/summaries/latest` | GET | **Riassunto Markdown giornaliero** generato da DeepSeek |
| `/api/portfolios/{code}/config` | POST | **Aggiornamento configurazione**: body `{"max_assets": N}` (intero ≥ 1); aggiorna il cap massimo asset del portafoglio e restituisce i parametri correnti |

### Metadati strumento (sector / country / area)

Il job batch `metadata` (cron, dopo il job `data`) popola su `instrument`:
- `sector`: settore merceologico per le azioni; per ETF/bond_yield è il comparto (`category` di yfinance).
- `country`: paese dal profilo yfinance (per molti ETF e i bond yield il campo è assente → `NULL`).
- `area`: `usa`, `eu`, `asia`, `emerging` o `other`, derivata dalla `country` con priorità **Emergenti > EU > USA > Asia > Altro**; quando la `country` manca si usa una mappatura manuale per simbolo (`app/area.py`).
- `metadata_date`: data dell'ultimo fetch (aggiornamento automatico dopo 30 giorni, forzabile con `--force`).

---

## 2. Webapp Angular (SPA)

Stack: **Angular 22 (standalone components)**, TypeScript, servito da `nginx`.

### 2.0 Change Detection con Signal (vincolo tecnico)

Il frontend usa la **change detection basata sui signal** di Angular (`signal()`,
`computed()`, `.set()`). Tutti i componenti che popolano la vista da chiamate HTTP
(`HttpClient.subscribe`) devono **memorizzare i dati in `signal`** e leggere i
valori nel template con le parentesi (`instruments()`).

Motivo: con Angular 22 (zoneless / change detection moderna), le semplici
assegnazioni a proprietà (`this.items = res.items`) non vengono rilevate dalla
vista: i dati arrivano dall'API ma il template non si aggiorna. Con i signal,
quando il valore cambia tramite `.set()`, Angular aggiorna automaticamente la
vista. **Regola per i futuri componenti**: nessuna proprietà mutata in callback
asincroni; usare sempre `signal` + `.set()` e `*ngFor="let x of items()"`.

### 2.1 Vista Principale del Martedì (Investment Thesis View)
La schermata chiave della webapp è la pagina di **Composizione Consigliata** (`/portfolios/:code/recommendations`), pensata per essere aperta il martedì prima di parlare con il promotore finanziario.

Ogni asset raccomandato è presentato sotto forma di **Scheda Tesi di Investimento (Investment Thesis Card)**:

```
┌────────────────────────────────────────────────────────────────────────┐
│ AAPL — Apple Inc. [ LONG ]                              Weight: 15.4%  │
├────────────────────────────────────────────────────────────────────────┤
│ 📊 Quant Score (Keras): +0.84       💬 DeepSeek Sentiment: Bullish     │
│ 💰 Prezzo Attuale: $228.50          🎯 Importo Target: €12,320 (54 pz) │
├────────────────────────────────────────────────────────────────────────┤
│ Tesi di Investimento (DeepSeek Rationale):                             │
│ "L'annuncio di nuovi acceleratori AI proprietari rafforza i margini    │
│ enterprise. Il sentiment di mercato è fortemente positivo e batte      │
│ le attese degli analisti sulle vendite trimestrali."                   │
├────────────────────────────────────────────────────────────────────────┤
│ Variazione vs settimana precedente: 🟢 +3.4%                           │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Altre Viste della SPA
1. **Dashboard (`/`)**: Panoramica di tutti i portafogli, stato dei job notturni (successo/fallimento) e data dell'ultimo aggiornamento dati.
2. **Portafoglio Attuale (`/portfolio`)**: Vista e **modifica manuale** delle posizioni effettivamente detenute (long/short). Tabella editor con `qty` e `avg_price` modificabili, toggle side, chiusura posizione e apertura di nuove posizioni dalla watchlist. Pulsante **"Applica Raccomandazioni del Modello"** (popola l'editor con la composizione target alla lettera) e pulsante **"SALVA"** che persiste via `POST /holdings/save`. Short rappresentato con `qty` negativa; P&L posizione = `qty × (close − avg_price)`.
3. **Tabella Segnali (`/portfolios/:code/signals`)**: Elenco completo di tutti gli strumenti del portafoglio con il dettaglio di come il punteggio matematico è stato corretto dal sentiment dell'IA.
4. **Notizie & Riassunti (`/portfolios/:code/news`)**: Visualizzatore Markdown formattato dei riassunti giornalieri creati da DeepSeek, con link diretti alle fonti originali di yfinance.
5. **Notizie Watchlist (`/news`)**: Vista a card compatte delle notizie recenti della watchlist, **raggruppate per settore** (`sector`), con badge sentiment colorato, filtri (periodo, settore, simbolo, "solo con sentiment") e paginazione incrementale ("Mostra altre notizie"). Ogni titolo è cliccabile.
6. **Dettaglio Notizia (`/news/:id`)**: Pagina di dettaglio di una singola notizia con titolo, editore, data, riassunto, analisi sentiment IA (con catalizzatore e rationale) e pulsante "Leggi la notizia originale" verso la fonte (es. finance.yahoo.com).
