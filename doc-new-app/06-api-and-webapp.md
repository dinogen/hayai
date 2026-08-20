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
| `/api/health` | GET | Stato del servizio e data dell'ultimo job in `job_run`. **Endpoint pubblico** (usato dal monitoraggio) |
| `/api/auth/login` | POST | Autenticazione (solo un utente). Body `{"username": ..., "password": ...}`; su successo imposta il cookie di sessione. **Pubblico** |
| `/api/auth/logout` | POST | Termina la sessione e cancella il cookie |
| `/api/auth/me` | GET | Stato sessione: `{"authenticated": true/false}`. **Pubblico** (mai 401) |
| `/api/portfolios` | GET | Elenco dei portafogli attivi |
| `/api/portfolios/{code}` | GET | Dettaglio del portafoglio (parametri, strumenti associati). Ogni strumento include anche `sector`, `country`, `area` e `metadata_date` (metadati dal job `metadata`). Filtro opzionale `?area=eu|usa|asia|emerging|other` per limitare gli strumenti a una specifica area geografica |
| `/api/portfolios/{code}/recommendations/latest` | GET | **Composizione consigliata (ultima data)**: pesi, side, importi, variazioni vs settimana precedente |
| `/api/portfolios/{code}/holdings` | GET | **Portafoglio attuale**: posizioni detenute (long/short), P&L, cash, NAV, watchlist (con `area`, `quant_score`, `llm_sentiment_modifier`, `final_signal`, `signal_date`, `vol_20`) e ultime raccomandazioni |
| `/api/portfolios/{code}/holdings/save` | POST | **Salvataggio portafoglio attuale**: applica lo stato desiderato (diff → `portfolio_trade` → snapshot posizioni → ricalcolo cash) |
| `/api/portfolios/{code}/holdings/report.md` | GET | **Report Markdown** delle posizioni correnti (prezzo di carico e data di acquisto), scaricabile dal bottone "Scarica Report MD" nella pagina Portafoglio |
| `/api/portfolios/{code}/watchlist` | GET | **Watchlist arricchita**: per ogni strumento espone `area`, `sector`, `current_price`, e l'ultimo segnale ibrido `quant_score`, `llm_sentiment_modifier`, `final_signal`, `signal_date` più `vol_20` (volatilità a 20 giorni dall'ultima predizione del modello). Strumenti senza segnale (es. bond yield fuori modello) restituiscono i valori segnale `null`. Ordinamento per `area`, poi `symbol` |
| `/api/portfolios/{code}/signals` | GET | Segnali ibridi (Quant Score + Sentiment IA) per strumento. Ogni segnale include `quant_score`, `llm_sentiment_modifier`, `final_signal`, `ai_rationale` e `sentiment_breakdown` (JSON con il dettaglio per-notizia che ha contribuito: `impact_score`, `impact_duration`, `confidence`, `age_hours`, `decay`, `contribution`) |
| `/api/portfolios/{code}/news` | GET | Notizie recenti collegate agli strumenti del portafoglio. Parametri opzionali: `?days=14` (retention, default 14), `?sector=` (filtro settore), `?symbol=` (filtro ticker), `?limit=50` (limite righe). Ogni notizia include `sector`, `area`, `impact_score`, `impact_duration`, `impact_surface`, `confidence` |
| `/api/news/{news_id}` | GET | Dettaglio di una singola notizia (titolo, publisher, data, summary, link originale, `impact_score`, `impact_duration`, `impact_surface`, catalyst e rationale IA) |
| `/api/instruments/{symbol}` | GET | **Dettaglio aggregato strumento**: meta (`sector`, `country`, `area`, `instrument_type`, `currency`), `latest_signal` (`quant_score`, `llm_sentiment_modifier`, `final_signal`, `signal_date`, `vol_20`), storico OHLCV `prices` (param `?days=`, default 250, max 750) e ultime 10 `news` con analisi IA. 404 se il simbolo non esiste; segnale `null` se non coperto dal modello |
| `/api/portfolios/{code}/summaries/latest` | GET | **Riassunto Markdown giornaliero** generato da DeepSeek |
| `/api/portfolios/{code}/config` | POST | **Aggiornamento configurazione**: body `{"max_assets": N}` (intero ≥ 1); aggiorna il cap massimo asset del portafoglio e restituisce i parametri correnti |
| `/api/markets/status` | GET | **Stato mercati globali**: per USA (NYSE/Nasdaq), Europe (Xetra/Euronext) e Asia (Tokyo TSE) restituisce `code`, `name`, `timezone`, `local_time`, `is_open`, `open_time`, `close_time`, `next_open_at`, `next_close_at`. Giorni feriali (lun-ven), senza calendario festività; DST gestito automaticamente via `zoneinfo` |

### Autenticazione e Sessione (cookie)

L'applicazione prevede **un solo utente** (proprietario). Le credenziali sono
fissate nel file `.env`:

- `AUTH_USERNAME` — nome utente
- `AUTH_PASSWORD` — password in chiaro (misura di sicurezza minima contro estranei)
- `AUTH_SESSION_SECRET` — stringa casuale lunga usata per firmare il cookie
  (generata con `python -c "import secrets; print(secrets.token_hex(32))"`)
- `AUTH_SESSION_MAX_AGE` — durata del cookie in secondi (default `43200` = 12 ore)

Meccanismo: **sessione via cookie firmato** (Starlette `SessionMiddleware`,
dipendenza `itsdangerous`), nessuno store lato server, nessun JWT. Il cookie
(`hayai_session`) è `HttpOnly`, `SameSite=Lax` e non usa il flag `Secure`
perché l'app gira in HTTP su LAN (Raspberry Pi).

- `POST /api/auth/login` confronta le credenziali con `hmac.compare_digest`
  e, in caso di successo, imposta la sessione.
- Tutti gli endpoint business (`/api/portfolios`, `/api/markets`, `/api/news`,
  `/api/instruments`, `/api/config`) richiedono la sessione: senza cookie il
  server risponde **401**.
- `/api/health` e `/api/auth/*` restano pubblici (il monitoraggio non deve
  autenticarsi; `/api/auth/me` non ritorna mai 401, serve al boot del frontend).

**Sviluppo (Angular dev server)**: per evitare il problema del cookie
`SameSite=Lax` su richieste cross-site (`localhost:4200` → `127.0.0.1:8000`),
il dev server usa un **proxy** (`web/proxy.conf.json`, abilitato in
`angular.json`): le richieste `/api` vengono inoltrate dal frontend al backend,
quindi per il browser tutto è **stessa origine** e la sessione funziona come in
produzione. `apiUrl` è relativo (`/api`) sia in dev sia in prod. Il backend
mantiene comunque CORS a origini esplicite con `allow_credentials=True` per
eventuali client che chiamino l'API direttamente in cross-origin.

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

### 2.1 Autenticazione nel Frontend

- **AuthService** (`core/services/auth.service.ts`): espone `login()`, `logout()`,
  `checkAuth()` (chiama `/api/auth/me`) e uno stato `authenticated` come `signal`.
- **Interceptor HTTP** (`core/interceptors/auth.interceptor.ts`): aggiunge
  `withCredentials: true` a ogni richiesta (necessario in dev cross-origin) e, su
  una risposta **401** da endpoint business, resetta lo stato e redirige a `/login`.
- **Guard di rotta** (`core/guards/auth.guard.ts`): tutte le rotte protette eseguono
  `checkAuth()`; se non autenticate redirigono alla pagina di login.
- **Pagina Login** (`features/login/login.component.ts`, rotta `/login`): form
  utente/password in stile "Cyber Light HUD"; su successo naviga alla dashboard.
- **Navbar**: mostra il pulsante "Esci" quando la sessione è attiva.

### 2.2 Vista Principale del Martedì (Investment Thesis View)
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

### 2.3 Altre Viste della SPA
1. **Dashboard (`/`)**: Panoramica di tutti i portafogli, stato dei job notturni (successo/fallimento), data dell'ultimo aggiornamento dati e box **"Mercati Aperti / Chiusi"** (USA, Europe, Asia) con pallino verde/rosso, ora locale e orari di borsa; lo stato è ricalcolato dal backend (`GET /api/markets/status`) e il frontend lo aggiorna ogni 60s.
2. **Portafoglio Attuale (`/portfolio`)**: Vista e **modifica manuale** delle posizioni effettivamente detenute (long/short). Tabella editor con `qty` e `avg_price` modificabili, toggle side, chiusura posizione e apertura di nuove posizioni dalla watchlist. Pulsante **"Applica Raccomandazioni del Modello"** (popola l'editor con la composizione target alla lettera) e pulsante **"SALVA"** che persiste via `POST /holdings/save`. Short rappresentato con `qty` negativa; P&L posizione = `qty × (close − avg_price)`.
3. **Tabella Segnali (`/portfolios/:code/signals`)**: Elenco completo di tutti gli strumenti del portafoglio con il dettaglio di come il punteggio matematico è stato corretto dal sentiment dell'IA. Ogni riga è espandibile e mostra il **dettaglio per-notizia** (`impact_score`, durata, confidenza, età, decay, contributo) che ha generato il modificatore.
4. **Watchlist (`/watchlist`)**: Tabella dell'intero universo con area geografica (badge colorato USA/EU/Asia/Emerging/Altro), ultimo segnale del modello (`quant_score`), ultimo coefficiente dalle news (`llm_sentiment_modifier`), segnale ibrido finale (`final_signal`), **volatilità a 20 giorni** (`vol_20`, colorata per livello di rischio: verde < 1.5%, giallo < 3%, rosso oltre) e prezzo corrente. Strumenti senza segnale mostrano `N/D` grigio. Riga es. bond yield `^TNX`: area, prezzo, `N/D` sui segnali. Ogni riga è **cliccabile** e apre il dettaglio strumento.
5. **Dettaglio Strumento (`/watchlist/:symbol`)**: Pagina raggiungibile cliccando una riga della Watchlist. Header con simbolo, nome, tipo, badge area, settore, paese, prezzo corrente e variazione % giorno. KPI box quantitativi (Quant Score, Sentiment Mod, Segnale Finale, Vol 20). **Candlestick chart** (libreria `lightweight-charts`) con istogramma volume, overlay MA20/MA50 e selettore periodo 3M/6M/1Y. Lista delle ultime 10 notizie dello strumento con badge `impact_score`, cliccabili verso il dettaglio notizia.
6. **Notizie & Riassunti (`/portfolios/:code/news`)**: Visualizzatore Markdown formattato dei riassunti giornalieri creati da DeepSeek, con link diretti alle fonti originali di yfinance.
7. **Notizie Watchlist (`/news`)**: Vista a card compatte delle notizie recenti della watchlist, **raggruppate per settore** (`sector`), con badge di `impact_score` colorato, durata attesa, filtri (periodo, settore, simbolo, "solo con analisi IA") e paginazione incrementale ("Mostra altre notizie"). Ogni titolo è cliccabile.
8. **Dettaglio Notizia (`/news/:id`)**: Pagina di dettaglio di una singola notizia con titolo, editore, data, riassunto, analisi IA (`impact_score`, durata, superficie di impatto, catalizzatore, confidenza e rationale) e pulsante "Leggi la notizia originale" verso la fonte (es. finance.yahoo.com).
