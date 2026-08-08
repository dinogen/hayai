# HAYAI v2 — Specifiche di Progetto (Clean Design)

Benvenuto nella documentazione della nuova applicazione **HAYAI v2**. 

Questa cartella (`doc-new-app/`) contiene la specifica architetturale e funzionale
completa per la riscrittura **da zero** del sistema. Il codice legacy precedente
è confinato nella root ed è considerato esclusivamente come archivio storico.

## Filosofia del Nuovo Sistema

Il sistema unisce l'approccio **Quant** (modelli matematici Keras per predire i rendimenti normalizzati per volatilità) con l'approccio **LLM/Qualitative** (API di **DeepSeek** per analizzare notizie, estrarre sentiment e generare la tesi di investimento in italiano). 

È ottimizzato per un esperimento personale da **€5.000** su un portafoglio unico (Azioni, ETF e Bond Yields, senza Forex) con approccio **Human-in-the-Loop**:
1. **Ogni notte (batch automatico)**: Aggiornamento dati di mercato (yfinance) + download notizie + analisi DeepSeek + inferenza modello + aggiustamento segnale + calcolo nuova composizione portafoglio long/short.
2. **Ogni martedì (revisione umana)**: Tu apri la webapp, analizzi la nuova composizione e leggi le motivazioni (tesi di investimento) generate da DeepSeek.
3. **Decisione e azione**: Discuti con il tuo promotore finanziario e decidi come aggiustare il portafoglio reale.

## Indice dei Documenti

| File | Titolo | Contenuto Principale |
|---|---|---|
| `00-index.md` | Indice e Visione | Panoramica, filosofia e mappa dei documenti |
| `01-target-architecture.md` | Architettura Target | Raspberry Pi 4/5, MariaDB, Cron, FastAPI, Angular, DeepSeek API (€5k experiment) |
| `02-database-schema.md` | Schema Database MariaDB | Tabelle, relazioni, indici, tracciamento NAV (`portfolio_cash`, `portfolio_position`) |
| `03-ml-pipeline.md` | Pipeline ML & Signal Hybrid | Training su PC (Jupyter), modelli per asset class, ONNX inference + Signal Adjustment |
| `04-news-llm-pipeline.md` | Notizie & DeepSeek LLM | Ingestione notizie yfinance, prompt strutturato JSON (sentiment, catalyst, rationale) |
| `05-portfolio-optimization.md` | Ottimizzazione Portafoglio | Pesi long/short, allocazione capitale €5.000 (90% investito, 10% cash) |
| `06-api-e-webapp.md` | API FastAPI & Webapp Angular | Endpoint REST in sola lettura, UI con schede tesi di investimento |
| `07-operativita-batch.md` | Operatività & Deploy Raspberry | Job CLI, pianificazione cron, guide di installazione nativa |
| `08-portfolio-lifecycle.md` | Ciclo di Vita del Portafoglio | Giorno 1 (bootstrap), evoluzione giornaliera, Mark-to-Market, gestione universo |
