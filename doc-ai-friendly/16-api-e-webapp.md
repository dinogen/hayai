# 16 — API (FastAPI) e webapp (Angular)

Questo documento definisce l'interfaccia **FastAPI** (backend REST) e la struttura
della **webapp Angular** (SPA statica servita da nginx) della nuova applicazione.

## 1. Architettura di servizio

```
Browser ──► nginx :80 (static Angular + reverse proxy)
                │
                ├─ /          → dist/ (SPA Angular)
                └─ /api/*     → uvicorn :8000 (FastAPI)
                                    │
                                    ▼
                                MariaDB (sola lettura per API)
```

- **FastAPI**: uvicorn, bind `127.0.0.1:8000`, reverse proxy via nginx.
- **Angular**: build di produzione → `/var/www/hayai` servito da nginx.
- **CORS**: configurato per l'origine servita (stesso dominio via proxy → nessun
  bisogno di CORS, ma si può abilitare per sviluppo).

## 2. API REST (FastAPI)

Tutte le risposte sono JSON. Le API sono **in sola lettura** (RF-55, RN-05); le
operazioni di scrittura avvengono solo nei job batch.

### 2.1 Health e meta

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/api/health` | GET | Stato servizio + data ultimo job (`job_run`) |
| `/api/models` | GET | Modelli registrati (id, versione, status, metriche) |

### 2.2 Portafogli e strumenti

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/api/portfolios` | GET | Elenco portafogli attivi con riepilogo |
| `/api/portfolios/{code}` | GET | Dettaglio portafoglio (strumenti, parametri, modello) |
| `/api/portfolios/{code}/instruments` | GET | Strumenti del portafoglio (simbolo, tipo, nome) |
| `/api/instruments` | GET | Ricerca/elenco strumenti (filtro tipo/attivo) |
| `/api/instruments/{symbol}` | GET | Dettaglio strumento |

### 2.3 Prezzi

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/api/instruments/{symbol}/prices` | GET | Serie prezzi (range date, limit) |
| `/api/portfolios/{code}/prices` | GET | Ultimi prezzi di tutti gli strumenti del portafoglio |
| `/api/portfolios/{code}/returns` | GET | Rendimenti periodici (1d/5d/20d) per strumento |

### 2.4 Predizioni

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/api/portfolios/{code}/predictions` | GET | Predizioni correnti (ultima data) per strumento |
| `/api/models/{id}/predictions` | GET | Storico predizioni di un modello |
| `/api/instruments/{symbol}/predictions` | GET | Serie predizioni di uno strumento |

### 2.5 Raccomandazioni

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/api/portfolios/{code}/recommendations/latest` | GET | Composizione consigliata (ultima data): peso, side, importo, variazione |
| `/api/portfolios/{code}/recommendations` | GET | Storico raccomandazioni (range date) |
| `/api/portfolios/{code}/recommendations/summary` | GET | Riepilogo long/short (totale pesi, conteggi) |

### 2.6 Notizie e riassunti

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/api/portfolios/{code}/news` | GET | Notizie del portafoglio (filtri data/simbolo) |
| `/api/instruments/{symbol}/news` | GET | Notizie di uno strumento |
| `/api/portfolios/{code}/summaries` | GET | Elenco riassunti markdown (data) |
| `/api/portfolios/{code}/summaries/{date}` | GET | Contenuto markdown del riassunto |

### 2.7 Esempio di contratto (raccomandazioni)

`GET /api/portfolios/medium_tech_usa/recommendations/latest`

```json
{
  "portfolio_code": "medium_tech_usa",
  "rec_date": "2026-08-08",
  "model": { "id": 3, "name": "multiasset_v1", "version": "2" },
  "equity_indicativa": 100000.0,
  "risk_percentage": 0.8,
  "items": [
    {
      "symbol": "AAPL",
      "instrument_type": "stock",
      "prediction": 1.24,
      "vol_20": 0.012,
      "weight": 0.154,
      "side": "long",
      "price": 228.5,
      "target_amount": 12320.0,
      "target_qty": 54,
      "prev_weight": 0.12
    }
  ]
}
```

### 2.8 Errori

- Formato coerente: `{"detail": "..."}` (standard FastAPI HTTPException).
- `404` risorse inesistenti, `400` parametri non validi, `503` servizio/DB non
  disponibile.

## 3. Webapp Angular

Stack: **Angular 17+ (standalone)**, TypeScript, build statico, servito da nginx.
Il frontend non ha accesso diretto al DB: chiama solo le API.

### 3.1 Struttura componenti

```
app/
├─ app.routes.ts            (routing)
├─ core/
│  ├─ services/
│  │  ├─ api.service.ts     (wrapper HttpClient per /api)
│  │  ├─ portfolio.service.ts
│  │  └─ auth.guard.ts      (opzionale, se autenticazione)
│  └─ models/               (interfacce TypeScript dei DTO)
├─ features/
│  ├─ dashboard/            → RF-50
│  ├─ portfolio-detail/     → RF-51
│  ├─ predictions/          → RF-52
│  ├─ recommendations/      → RF-53
│  ├─ news/                 → RF-54 (riassunti markdown)
│  └─ shared/               (tabelle, spinner, formatter)
```

### 3.2 Viste principali

1. **Dashboard** (`/`): card portafogli (ultimo aggiornamento, n strumenti,
   data predizioni, stato ultimi job), link alle sezioni.
2. **Dettaglio portafoglio** (`/portfolios/:code`): tabella strumenti (simbolo,
   tipo, prezzo, variazione, predizione), parametri del portafoglio.
3. **Predizioni** (`/portfolios/:code/predictions`): tabella con predizione
   normalizzata e segnale (color coding), ordinabile/filtrabile.
4. **Composizione consigliata** (`/portfolios/:code/recommendations`): tabella
   long/short con peso, importo indicativo, variazione vs precedente; riepilogo
   totale long/short.
5. **Notizie** (`/portfolios/:code/news`): elenco notizie e **render del markdown**
   dei riassunti (componente markdown).
6. **Stato sistema** (footer/`/health`): data ultimo job per tipo, esito.

### 3.3 Rendering markdown

- Libreria di rendering markdown (es. `marked` + `dompurify` per la sicurezza
  del rendering del contenuto `news_summary.markdown`).
- Layout "documento" per i riassunti, con breadcrumb portafoglio → data.

### 3.4 Build e deploy

- Build su PC (o CI): `ng build --configuration production` → `dist/browser`.
- Copia su Raspberry in `/var/www/hayai`.
- nginx: `server { root /var/www/hayai; location /api { proxy_pass http://127.0.0.1:8000; } location / { try_files $uri $uri/ /index.html; } }`.

## 4. Autenticazione (opzionale)

- Se l'esposizione supera la LAN: **basic auth** su nginx o **token** (FastAPI
  dependency) per l'accesso esterno.
- Le API di amministrazione (lancio job) non sono esposte via webapp (si usano
  CLI/cron); eventualmente solo endpoint di trigger protetti.

## 5. Requisiti soddisfatti

- RF-50/51/52/53/54/55/56 → §3.2-3.4.
- RN-03 (performance) → query indicizzate, paginazione (`limit/offset`), frontend
  statico.
- RN-05 (sicurezza) → API in sola lettura, proxy, bind loopback.

## 6. Note di implementazione

- Usare **Pydantic models** per i DTO di risposta (coerente con OpenAPI).
- Query con `SELECT` su `price_daily`/`prediction`/`recommendation` indicizzate
  sulle chiavi naturali (doc `13`).
- Paginare le serie lunghe (default `limit=1000`).
- Tenere separati i modelli API dai modelli DB (niente ORM esposto direttamente).
