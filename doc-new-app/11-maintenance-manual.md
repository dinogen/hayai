# 11 — Manuale di Manutenzione (Manuale Operativo)

Manuale sintetico per chi mantiene HAYAI v2. Descrive lo scopo del sistema, i file
`.bat` di avvio, le tabelle principali e il flusso dei dati, i job batch e la loro
schedulazione, il modello ML e l'uso della webapp.

---

## 1. Scopo Generale del Programma

**HAYAI v2** è un "Personal Quant": un sistema di supporto decisionale che ogni notte
analizza un portafoglio di test da **€5.000** (simulazione, nessun denaro reale) e
produce una **composizione consigliata long/short** con relative **tesi di investimento
in italiano**.

Combina due approcci:

1. **Quant**: un modello di rete neurale (Keras, esportato in ONNX) predice i rendimenti
   futuri normalizzati per volatilità → `quant_score`.
2. **LLM**: DeepSeek analizza le notizie del giorno e produce sentiment + rationale →
   `llm_sentiment_modifier`.

L'utente apre la webapp **ogni martedì** (revisione umana con il promotore finanziario),
legge le schede tesi e decide se applicare o meno le raccomandazioni. **Nessun ordine
automatico**: il sistema produce solo indicazioni.

### Componenti principali
| Componente | Ruolo |
|---|---|
| **MariaDB** (`hayai`) | Database unico di prezzi, notizie, segnali e stato portafoglio |
| **Batch Python CLI** | Job notturni (`python -m app.cli <job>`) schedulati via cron |
| **yfinance** | Fonte dati prezzi OHLCV, indici e notizie |
| **DeepSeek API** | Analisi semantica notizie (sentiment + rationale) |
| **FastAPI + uvicorn** | Backend REST in sola lettura (su `127.0.0.1:8000`) |
| **Angular SPA + nginx** | Webapp di consultazione (Dashboard, Segnali, Recommendations, Notizie) |
| **Keras/ONNX** | Modello quant addestrato su PC, inferenza leggera sul Pi |

---

## 2. I File Batch (`.bat`)

Sono gli script di avvio manuale per **Windows** (l'ambiente di sviluppo/test; sul
Raspberry Pi la stessa logica è eseguita da cron). Tutti i file sono nella root del
progetto (tranne `train_v3.bat`, dentro `hayai-new/`).

| File | Cosa fa | Quando lanciarlo e perché |
|---|---|---|
| `avvia_mariadb_server.bat` | Avvia il server MariaDB in console (`mariadbd.exe`) | **Sempre per primo** (dev/test): senza DB nulla funziona. Da lanciare una sola volta, restare aperto |
| `avvia_backend.bat` | Avvia il backend FastAPI (uvicorn, `--reload`) su `127.0.0.1:8000` | Prima di usare la webapp in dev. Serve l'API per qualsiasi vista dell'app |
| `avvia_frontend.bat` | Avvia il dev server Angular su `http://localhost:4200` | In sviluppo, per consultare la SPA (richiede backend avviato) |
| `avvia_ciclo_completo.bat` | Esegue la **pipeline notturna completa**: `data → news → sentiment → predict → signal → recommend → summaries` | Per simulare il ciclo notturno a mano (es. quando il Pi non gira, dopo un fermo di più giorni, o per test). Sostituisce il cron su Windows |
| `scarica_dati.bat` | **Identico al ciclo completo** (stessi 7 job), versione con `--portfolio main` esplicito e gestione errori con `pause` | Alternativa storica ad `avvia_ciclo_completo.bat` per il ripristino dopo downtime |
| `avvia_prediction.bat` | Esegue solo la coda di segnale: `predict → signal → recommend` | Quando i prezzi/notizie sono già aggiornati e si vogliono **ricalcolare solo segnali e composizione** (es. dopo aver riattivato `NEWS_LLM_ENABLED`, o per rigenerare le raccomandazioni del martedì) |
| `train_universe.bat` | Ingestion dati (100 asset, 5 anni) + **training** del modello MLP | Raramente, **ogni 2-3 mesi**, per riaddestrare il modello con dati più recenti |
| `hayai-new\train_v3.bat` | Training completo di una versione modello (`v3`): seed universo → download storico 5y → training → `verify` → `backtest` | Quando si sperimenta una nuova versione del modello prima di renderla attiva |

**Ordine tipico di avvio in dev**: `avvia_mariadb_server.bat` → `avvia_ciclo_completo.bat`
(oppure `scarica_dati.bat`) → `avvia_backend.bat` → `avvia_frontend.bat`.

> **Nota**: i batch attivano il venv (`venv\Scripts\activate`) e impostano
> `PYTHONPATH=hayai-new`. In produzione (Raspberry Pi) questi script non servono:
> l'esecuzione è affidata a cron (vedi §4).

---

## 3. Tabelle Principali e Flusso dei Dati

### 3.1 Mappa sintetica delle tabelle (MariaDB `hayai`)
| Tabella | Contenuto | Scritta da |
|---|---|---|
| `portfolio` | Parametri del portafoglio (capitale €5.000, `n_long`, `n_short`, `max_assets`, `risk_percentage`) | setup / webapp (config) |
| `instrument` | Universo degli strumenti (`symbol`, tipo `stock/etf/bond_yield`, `sector`, `country`, `area`) | setup, job `metadata` |
| `portfolio_instrument` | Associazione portafoglio ↔ strumenti (watchlist) | setup |
| `price_daily` | Prezzi OHLCV giornalieri per strumento | job `data` (upsert) |
| `news` | Notizie scaricate da yfinance (idempotente su `source_id`) | job `news` |
| `news_sentiment` | Analisi DeepSeek per notizia (`impact_score`, durata, superficie, `confidence`, `rationale`) | job `sentiment` |
| `model_registry` | Modelli ONNX registrati (`active`/`draft`), feature, min/max | deploy/training |
| `model_prediction` | Output inferenza ONNX (`prediction` = quant_score, `vol_20`) | job `predict` |
| `portfolio_signal` | Segnale ibrido (`quant_score` + `llm_sentiment_modifier` = `final_signal`, `ai_rationale`, breakdown) | job `signal` |
| `portfolio_recommendation` | Composizione consigliata (pesi, side, `target_amount`, `target_qty`) | job `recommend` |
| `portfolio_trade` | Log operazioni manuali (`buy/sell/short/cover`) | webapp "Portafoglio Attuale" |
| `portfolio_position` | Posizioni detenute giorno per giorno (`qty`, `avg_price`, `market_value`; short = `qty` negativa) | webapp + job `nav` (mark-to-market) |
| `portfolio_cash` | Liquidità giornaliera | webapp (operazioni) |
| `news_summary` | Riassunto Markdown giornaliero notizie+tesi | job `summaries` |
| `job_run` | Log di ogni esecuzione job (status `running/success/failed`) | CLI (automatico) |

### 3.2 Flusso dei dati (pipeline notturna)
```
 yfinance ──► price_daily ──► feature (pannello) ──► ONNX inferenza ──► model_prediction
      │                                                                      │
      │                                                                      ▼
      └──► news ──► DeepSeek API ──► news_sentiment ──► portfolio_signal (final_signal = quant + llm)
                                                                             │
                                                                             ▼
                                                        portfolio_recommendation (pesi long/short €4.500)
                                                                             │
                                              (operazioni manuali webapp)    ▼
                                              portfolio_trade ──► portfolio_position + portfolio_cash
                                                                             │
                                                              job nav ──► mark-to-market giornaliero
                                              news_summary (riassunto Markdown per la webapp)
```
In sintesi: **prezzi e notizie → segnali → raccomandazioni → stato portafoglio**.
Il database è l'unico punto di scambio tra batch, API e webapp.

---

## 4. Job Batch e Schedulazione

Tutti i job si lanciano con `python -m app.cli <nome>` (da `hayai-new/`, venv attivo).
Ogni esecuzione viene registrata in `job_run` con stato e durata.

| Job | Scopo | Schedulazione cron (Pi) |
|---|---|---|
| `data` | Scarica prezzi OHLCV da yfinance (`period="1y"`, **upsert idempotente** → recupera automaticamente i giorni mancanti dopo un fermo) | Lun-Ven 02:15 |
| `metadata` | Aggiorna `sector`/`country`/`area` da yfinance (salta se aggiornato < 30gg; `--force` per forzare) | Lun-Ven 02:30 |
| `news` | Scarica e upsert delle notizie yfinance per gli strumenti attivi | Lun-Ven 02:45 |
| `sentiment` | Analizza le nuove notizie con DeepSeek (`impact_score`, `rationale`). **Salta se `NEWS_LLM_ENABLED=false`** (stato `disabled`, zero token consumati) | Lun-Ven 03:00 |
| `predict` | Calcola le feature, inferenza ONNX e salva `quant_score` in `model_prediction` | Lun-Ven 03:15 |
| `signal` | Combina `quant_score` + modificatore DeepSeek → `final_signal` in `portfolio_signal` | Lun-Ven 03:30 |
| `recommend` | Seleziona top `n_long` / bottom `n_short`, calcola pesi, `target_amount`/`target_qty` su €4.500 | Lun-Ven 03:45 |
| `nav` | Mark-to-Market giornaliero: rivaluta le posizioni ai prezzi correnti (senza modificare quantità) | Lun-Ven 03:50 |
| `summaries` | Genera il riassunto Markdown (notizie + tesi) in `news_summary` | Lun-Ven 04:00 |
| `cleanup` | Elimina notizie/sentiment oltre i 14 giorni e cache parquet scadute (`--days` configurabile) | Lun-Ven 04:30 |
| `verify` | **Manuale**: valuta il modello deployato sul dataset attuale (metriche, drift) | non in cron |
| `backtest` | **Manuale**: backtest della selezione long/short sul test set | non in cron |

La sequenza è **rigorosa**: ogni job dipende da quelli precedenti (`data → sentiment →
predict → signal → recommend → nav → summaries`). Il cron li sfasa di 15 minuti per
evitare sovrapposizioni. Il backup notturno (`scripts/backup.sh`) gira alle 04:15.

> **Cron di produzione**: solo Lun-Ven (`* * * * 1-5`). Il job `data` scarica comunque
> un anno di storico, quindi lunedì recupera i weekend e gli eventuali giorni di fermo.

---

## 5. Il Modello in Sintesi

- **Architettura**: MLP densa `100 → 80 → 20 → 1 (sigmoid)`, loss MSE, early stopping.
- **Training**: sul **PC** in Jupyter, **ogni 2-3 mesi** (non ogni giorno). Esportato in
  **ONNX** insieme a `mins.csv`/`maxs.csv`/`config.json` e registrato in `model_registry`.
- **Versione di produzione**: `stock_model v2` (**24 feature**, "type-agnostic"):
  - 12 base: momentum (5/10/20gg), volatilità (10/20gg), `vol_ratio`, `zscore_20`,
    `trend_50`, `vol_regime`, `mom_vol_adj`, `volume_shock`, `log_return`;
  - 6 cross-sezionali (rank nell'universo, momentum relativo vs SPY, `excess_ret_5`, `beta_20`);
  - 6 di regime di mercato (rendimenti SPY 5/20gg, `breadth_20`, `dispersion_20`).
  - Preprocessing: **winsorizzazione** ai percentili 0.5/99.5, min-max scaling.
- **Target**: `clip(ln(close_{t+5}/close_t) / vol_20, clip_min, clip_max)` — rendimento a
  5 giorni normalizzato per volatilità.
- **Inferenza notturna** (`predict`): feature sul pannello completo → ONNX (`onnxruntime`,
  leggero su ARM) → denormalizzazione → `quant_score` in `model_prediction`.
- **Segnale ibrido**: `final_signal = quant_score + llm_sentiment_modifier`, con il
  modificatore DeepSeek limitato a ±0.20 e derivato da `impact_score`, confidenza e
  decadimento temporale della notizia (brief 24h / medium 96h / long 336h).
- **Verifica**: il job `verify` riproduce lo split, calcola RMSE/MAE/R²/hit-rate e
  segnala eventuale drift dei min/max. Il job `backtest` valuta l'edge della selezione
  long/short (Spearman, spread cumulato). Le versioni sperimentali v1/v3 restano
  archiviate in `model_registry`; **v2 è quella attiva**.

> **Nota interpretativa**: l'edge cross-sezionale del quant è debole (il backtest ha
> mostrato un buon tilt long-only, short ≈ rumore). Il modello va trattato come **input
> debole dell'ibrido**: la parte qualitativa DeepSeek è spesso quella che decide.

---

## 6. La Webapp e Come Usare le Informazioni

Webapp Angular servita da nginx (dev: `http://localhost:4200`), backend su
`http://127.0.0.1:8000` (Swagger: `/docs`). API in sola lettura tranne i salvataggi
manuali di portafoglio/config.

| Vista | URL | Cosa mostra e come usarla |
|---|---|---|
| **Dashboard** | `/` | Panoramica portafogli, stato ultimi job notturni, data ultimo aggiornamento, box "Mercati Aperti/Chiusi" (USA/EU/Asia, aggiornato ogni 60s) |
| **Composizione Consigliata** (vista del martedì) | `/portfolios/:code/recommendations` | **Le Schede Tesi di Investimento**: per ogni asset raccomandato peso, side, Quant Score, sentiment DeepSeek, prezzo, importo target e quote, **rationale in italiano** e variazione vs settimana precedente. È la pagina da aprire prima della revisione con il promotore |
| **Portafoglio Attuale** | `/portfolio` | Editor delle posizioni reali (qty, avg_price, side long/short). Qui si registrano le operazioni fatte (compri/vendi), con pulsante "Applica Raccomandazioni del Modello" per allineare alla target e "SALVA" (→ `portfolio_trade`, ricalcolo cash) |
| **Tabella Segnali** | `/portfolios/:code/signals` | Tutti gli strumenti con `quant_score`, modificatore LLM, `final_signal` e il **breakdown per-notizia** (quale notizia ha contribuito quanto, con decay) |
| **Watchlist** | `/watchlist` | Universo completo con area (badge), segnali, `vol_20` (colorata per rischio) e prezzo. Righe cliccabili verso il dettaglio strumento |
| **Dettaglio Strumento** | `/watchlist/:symbol` | Header informativo, KPI (Quant/Sentiment/Finale/Vol), **candlestick chart** con MA20/MA50 e le ultime 10 notizie con analisi IA |
| **Notizie & Riassunti** | `/portfolios/:code/news` | Riassunto Markdown giornaliero generato da DeepSeek con link alle fonti originali |
| **Notizie Watchlist** | `/news` | Card notizie raggruppate per settore, badge `impact_score`, filtri e paginazione |
| **Dettaglio Notizia** | `/news/:id` | Titolo, editore, riassunto, analisi IA completa (impatto, durata, superficie, catalyst, confidenza, rationale) e link all'originale |
| **Configurazione** | `/config` | Capitale iniziale simulato, `max_assets`, e **Reset Portafoglio** (azzera posizioni/cash senza toccare i dati del modello) |

**Flusso d'uso del martedì**: aprire *Composizione Consigliata* → leggere le schede tesi
e i segnali → valutare con il promotore → registrare in *Portafoglio Attuale* le operazioni
effettivamente eseguite. Nei giorni successivi il job `nav` valuta le posizioni ai prezzi
di mercato e l'header mostra NAV e P&L (vs mese scorso e vs €5.000 iniziali).

---

## 7. Procedura di Deploy e Aggiornamento sul Raspberry Pi

Dopo aver apportato modifiche al codice (backend Python o frontend Angular) ed eseguito il trasferimento o l'aggiornamento sul Raspberry Pi, occorre seguire questa procedura per aggiornare l'ambiente di produzione:

### 7.1 Aggiornamento del Frontend (Angular SPA)
1. Spostarsi nella directory del frontend:
   ```bash
   cd ~/hayai/hayai-new/web
   ```
2. Installare eventuali nuove dipendenze (se necessario):
   ```bash
   npm install
   ```
3. Eseguire la build di produzione:
   ```bash
   npm run build
   ```
   * I file statici compilati vengono generati nella directory `dist/web` (o `dist/web/browser`), configurata come root di **nginx**.

### 7.2 Aggiornamento del Backend (FastAPI)
1. Spostarsi nella directory del progetto:
   ```bash
   cd ~/hayai/hayai-new
   ```
2. Attivare il virtual environment Python e aggiornare le dipendenze (se `requirements.txt` è stato modificato):
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Riavviare il servizio systemd di FastAPI:
   ```bash
   sudo systemctl restart hayai-api
   ```
4. Controllare lo stato del servizio per verificare che sia attivo e in ascolto:
   ```bash
   sudo systemctl status hayai-api
   ```

### 7.3 Database e Script Batch
- Eventuali modifiche strutturali al database (es. nuove colonne) devono essere applicate su MariaDB prima o contestualmente all'aggiornamento del backend.
- Gli script batch CLI (`python -m app.cli <job>`) non richiedono riavvii di demoni poiché ogni esecuzione via `cron` carica l'ultima versione del codice sorgente.

---

## 8. Checklist di Primo Soccorso

| Sintomo | Cosa controllare |
|---|---|
| Nessun dato nuovo | `job_run` per lo stato dei job; log in `logs/cron.log` e `logs/hayai.log` |
| Job `data` fallito | Connessione internet/`yfinance` (rate limit); i dati mancanti vengono recuperati al run successivo (upsert). Il client yfinance condiviso (`app/yf_client.py`) ritenta automaticamente con backoff esponenziale su HTTP 429/5xx e su risposte vuote; se il blocco persiste oltre i retry, il job logga l'errore e il ciclo continua |
| Job `metadata` con `429 Too Many Requests` | Rate limit Yahoo su `quoteSummary`; il client condiviso ritenta con backoff. Se fallisce ancora, i metadati restano quelli esistenti (`metadata_date` invariato) e vengono ritentati al prossimo run |
| Nessuna raccomandazione | Verificare che `predict`/`signal` siano andati a buon fine; `model_registry` abbia un modello `active` |
| Sentiment assente | `NEWS_LLM_ENABLED` (`.env` o Configurazione) e credenziali DeepSeek |
| NAV fermo | Il job `nav` gira solo se il portafoglio ha posizioni; cash/posizioni si muovono solo con operazioni manuali in webapp |
| Webapp vuota in dev | Avviare `avvia_mariadb_server.bat`, poi `avvia_backend.bat`, poi `avvia_frontend.bat` |
