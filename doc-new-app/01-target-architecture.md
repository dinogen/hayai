# 01 — Architettura Target (Versione Semplificata per Sperimentazione Personale)

Questo documento descrive l'architettura aggiornata e semplificata per HAYAI v2,
ottimizzata per un **portafoglio unico personale** e un capitale di **€5.000**,
ospitata su un **Raspberry Pi** con MariaDB, FastAPI, Angular e DeepSeek.

---

## 1. Filosofia del Progetto "Personal Quant"

Gestire un portafoglio di test da €5.000 su un unico portafoglio misto (Azioni, ETF e Rendimenti Obbligazionari, senza Forex) ci permette di **tagliare drasticamente la complessità** mantenendo tutta l'efficacia del sistema Quant + AI.

### Cosa cambia rispetto all'idea multi-portafoglio:
- **Un solo portafoglio attivo**: Niente suddivisioni regionali o multiple. Un'unica watchlist di strumenti (es. 20-30 asset tra azioni, ETF e bond yields).
- **Niente Forex**: Eliminato il mercato valutario e le relative conversioni (tutto in EUR o USD nativo).
- **Capitale di riferimento**: €5.000 (il dimensionamento dei lotti e le raccomandazioni si basano su questa cifra).
- **Focus sull'esperimento**: Massima pulizia del codice, facilità di debug e totale controllo visivo dalla webapp.

---

## 2. Componenti dell'Architettura Semplificata

```
                    ┌──────────────────────────────────────────────┐
                    │              RASPBERRY PI                    │
                    │                                              │
  Internet ──► yfinance (prezzi OHLCV, indici, notizie)          │
           ──► DeepSeek API (Sentiment & Rationale)                │
                    │                                              │
                    │  ┌───────────┐   cron    ┌───────────────┐  │
                    │  │ BATCH PY  │◄──────────│  cronie       │  │
                    │  └─────┬─────┘           └───────────────┘  │
                    │        │ read/write                          │
                    │        ▼                                    │
                    │  ┌─────────────────────────┐                │
                    │  │      MariaDB            │                │
                    │  │ (prices, news, signals, │                │
                    │  │  portafoglio unico)     │                │
                    │  └───────────┬─────────────┘                │
                    │              │                              │
                    │              ▼                              │
                    │  ┌────────────────────┐  ┌───────────────┐  │
                    │  │  FastAPI (uvicorn) │  │ nginx (static │  │
                    │  │  REST API /api     │  │ + reverse pr.)│  │
                    │  └─────────┬──────────┘  └───────┬───────┘  │
                    │            │                     │          │
                    │            └─────────┬───────────┘          │
                    │                      ▼                      │
                    │              Angular SPA (dist)             │
                    └──────────────────────────────────────────────┘
```

1. **Database MariaDB**: Semplificato (rimossa la tabella forex, un solo portafoglio principale).
2. **Batch Cron**: Script Python leggeri che eseguono il ciclo notturno in pochi secondi.
3. **Backend FastAPI**: API REST ultra-rapide per servire il portafoglio unico.
4. **Frontend Angular**: Dashboard pulita, visualizzazione dei segnali ibridi e **Schede Tesi di Investimento (DeepSeek)** per la revisione del martedì.

---

## 3. Il Ciclo Operativo Settimanale

1. **Ogni Notte (Batch)**:
   - Aggiornamento prezzi (azioni, ETF, bond yields come `^TNX`).
   - Download notizie yfinance e analisi DeepSeek (Sentiment + Rationale in italiano).
   - Inferenza ONNX (Modello Keras) + Modulatore di Sentiment = **Segnale Ibrido**.
   - Calcolo dei pesi target long/short per i €5.000.
2. **Ogni Martedì (Revisione Umana)**:
   - Apri la webapp dal browser.
   - Leggi le variazioni proposte e le motivazioni di DeepSeek.
   - Parli con il tuo promotore ed esegui i cambi sul conto reale.
