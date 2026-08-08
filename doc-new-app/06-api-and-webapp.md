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
| `/api/portfolios/{code}` | GET | Dettaglio del portafoglio (parametri, strumenti associati) |
| `/api/portfolios/{code}/recommendations/latest` | GET | **Composizione consigliata (ultima data)**: pesi, side, importi, variazioni vs settimana precedente |
| `/api/portfolios/{code}/signals` | GET | Segnali ibridi (Quant Score + Sentiment IA) per strumento |
| `/api/portfolios/{code}/news` | GET | Notizie recenti collegate agli strumenti del portafoglio |
| `/api/portfolios/{code}/summaries/latest` | GET | **Riassunto Markdown giornaliero** generato da DeepSeek |

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
2. **Tabella Segnali (`/portfolios/:code/signals`)**: Elenco completo di tutti gli strumenti del portafoglio con il dettaglio di come il punteggio matematico è stato corretto dal sentiment dell'IA.
3. **Notizie & Riassunti (`/portfolios/:code/news`)**: Visualizzatore Markdown formattato dei riassunti giornalieri creati da DeepSeek, con link diretti alle fonti originali di yfinance.
