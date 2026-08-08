# 12 — Requisiti funzionali e non funzionali

Questo documento raccoglie i **requisiti** della nuova applicazione (batch + webapp
su Raspberry). Ogni requisito è identificato da un codice (`RF` funzionale,
`RN` non funzionale) così da poter essere tracciato nel nuovo progetto.

## 1. Attori

| Attore | Descrizione |
|---|---|
| **Utente** | Consulta la webapp: portafogli, predizioni, raccomandazioni, notizie |
| **Scheduler (cron)** | Esegue i job batch secondo pianificazione |
| **Amministratore** | Configura portafogli, universi, modello, pianificazioni |

## 2. Requisiti funzionali

### 2.1 Gestione portafogli e strumenti

- **RF-01** — Il sistema gestisce più portafogli, ciascuno con un proprio nome/id.
- **RF-02** — Ogni portafoglio contiene strumenti di tipo **azione**, **ETF**,
  **valuta (fx)** e **rendimento obbligazionario** (bond yield).
- **RF-03** — Ogni strumento ha: simbolo yfinance, nome, tipo, valuta, settore/paese
  (solo per azioni/ETF, opzionale) e flag `active`.
- **RF-04** — L'amministratore aggiunge/rimuove strumenti dai portafogli
  (CRUD via CLI o endpoint admin).
- **RF-05** — Un **universo** opzionale (strumenti candidati) alimenta il training
  del modello.

### 2.2 Aggiornamento dati (batch)

- **RF-10** — Un job **daily** aggiorna i prezzi OHLCV di tutti gli strumenti attivi
  da yfinance (download batch) e li salva in `price_daily` (upsert idempotente).
- **RF-11** — Il job aggiorna anche le serie **forex** e **indici** configurati.
- **RF-12** — Strumenti senza dati recenti (es. errore download) vengono registrati
  nel `job_run` senza interrompere il job.
- **RF-13** — Il job è riavviabile senza duplicare dati (upsert su `date+symbol`).

### 2.3 Notizie e riassunti markdown (batch)

- **RF-20** — Un job **daily** scarica le notizie finanziarie relative ai simboli dei
  portafogli (yfinance `Ticker.news` / `yf.Search`).
- **RF-21** — Le notizie vengono **deduplicate** (chiave: id notizia yfinance) e
  salvate in `news`.
- **RF-22** — Il sistema genera un **riassunto in markdown** per portafoglio e data:
  elenco titoli, editore, data, link, ticker correlati (vedi `15-pipeline-notizie.md`).
- **RF-23** — I riassunti vengono salvati in `news_summary` e scritti anche come
  file `.md` su disco (export).
- **RF-24** — Le notizie possono essere filtrate per portafoglio, strumento e data.

### 2.4 Modello e predizioni

- **RF-30** — Il modello viene addestrato **su PC** in un notebook Jupyter partendo
  dai dati del database (o da esport CSV/parquet).
- **RF-31** — L'artefatto del modello comprende: rete (ONNX + `.keras`), parametri
  di normalizzazione (min/max per colonna), `label_min/label_max`, e **metadati**
  (id modello, versione, data training, metriche, fingerprint dataset).
- **RF-32** — Il modello viene registrato in `model_registry` (id+versione) e i
  file sono deployati sul Raspberry.
- **RF-33** — Un job **predict** applica il modello all'ultima data disponibile per
  ogni strumento e salva le **predizioni** in `prediction` (una riga per
  strumento/data).
- **RF-34** — Le predizioni sono denormalizzate e clippate ai range del modello
  (coerente con doc `04`).

### 2.5 Raccomandazioni long/short (batch)

- **RF-40** — Un job **recommend** calcola i **pesi target** a partire dalle
  predizioni: `peso = prediction.clip / vol_20`, selezione top `n_long` long e
  bottom `n_short` short, normalizzazione somma |pesi| = 1 (vedi doc `05`).
- **RF-41** — Le raccomandazioni vengono salvate in `recommendation` per
  portafoglio e data, con indicazione del senso (long/short) e del peso.
- **RF-42** — Il sistema calcola una **posizione indicativa di massima**
  (importo consigliato = `equity_indicativa * risk_percentage`, quantità =
  `round(importo / prezzo)`), **senza** generare ordini reali.
- **RF-43** — Parametri di raccomandazione (`n_long`, `n_short`,
  `risk_percentage`, `qty_diff_perc_min`, soglie) configurabili per portafoglio.

### 2.6 Webapp

- **RF-50** — Dashboard: riepilogo portafogli, ultimi prezzi, data ultimo
  aggiornamento, stato job.
- **RF-51** — Dettaglio portafoglio: strumenti, prezzi, rendimenti recenti,
  predizioni correnti.
- **RF-52** — Pagina **predizioni**: tabella strumenti con predizione normalizzata
  e segnale.
- **RF-53** — Pagina **composizione consigliata**: pesi long/short, senso, importo
  indicativo, differenza vs composizione precedente.
- **RF-54** — Pagina **notizie/riassunti**: riassunti markdown renderizzati per
  portafoglio/data, filtro per strumento.
- **RF-55** — La webapp legge i dati tramite **API FastAPI** in sola lettura.
- **RF-56** — L'applicazione è consultabile da LAN; l'accesso da internet avviene
  solo attraverso canali sicuri (VPN/tunnel) o con autenticazione.

### 2.7 Amministrazione e operazioni

- **RF-60** — Un'interfaccia CLI (`python -m app <job> ...`) esegue ogni job batch
  manualmente, oltre che via cron.
- **RF-61** — Ogni esecuzione di job viene registrata in `job_run` (esito, durata,
  errori, conteggi).
- **RF-62** — Alert opzionale su errore (es. Telegram) configurabile.

## 3. Requisiti non funzionali

- **RN-01 (Idempotenza)** — Rieseguire un job non produce duplicati né
  doppie contabilizzazioni (upsert su chiavi naturali).
- **RN-02 (Riproducibilità)** — Stesso dataset + stesso artefatto modello →
  stesse predizioni; artefatti e parametri versionati.
- **RN-03 (Performance)** — I job notturni completano in finestre temporali
  compatibili con Raspberry Pi 4/5 (stima: poche centinaia di strumenti → minuti);
  la webapp risponde < 2s sulle query principali.
- **RN-04 (Disponibilità)** — La webapp è servita dal Raspberry; il DB è locale.
  Un crash di un job non deve corrompere lo stato (transazioni, upsert).
- **RN-05 (Sicurezza)** — Nessuna credenziale in repo; `.env` non versionato; DB in
  ascolto solo su loopback; API in sola lettura; niente credenziali broker.
- **RN-06 (Osservabilità)** — Log strutturati per job; `job_run` in DB; metriche di
  base (durata, errori).
- **RN-07 (Manutenibilità)** — Logica batch pura e testabile; test automatici
  obbligatori per calcolo pesi, dedup notizie, upsert.
- **RN-08 (Scalabilità futura)** — L'aggiunta di un secondo provider dati o di
  un'elaborazione notizie più sofisticata non deve richiedere riscritture.
- **RN-09 (Backup)** — Backup giornaliero del DB (`mysqldump`) e degli artefatti.

## 4. User stories principali

1. Come utente, ogni mattina apro la webapp e vedo le **predizioni** aggiornate per
   i miei portafogli.
2. Come utente, vedo la **composizione consigliata** long/short con i pesi e gli
   importi indicativi, così so cosa eventualmente riequilibrare manualmente.
3. Come utente, leggo il **riassunto in markdown** delle notizie del giorno per ogni
   portafoglio.
4. Come amministratore, aggiungo un ETF obbligazionario a un portafoglio e al job
   serale successivo i dati sono aggiornati.
5. Come amministratore, addestro un nuovo modello su PC, lo registro con una versione
   e lo rendo attivo senza toccare il resto del sistema.
6. Come amministratore, lancio manualmente un job e ne verifico l'esito in `job_run`.

## 5. Criteri di accettazione (esempi)

- **ACC-01** — Dopo `job data`, per ogni strumento attivo esiste una riga in
  `price_daily` alla data odierna (o errore tracciato in `job_run`).
- **ACC-02** — Rieseguendo `job data` lo stesso giorno, il numero di righe in
  `price_daily` non aumenta (idempotenza).
- **ACC-03** — `job news` + `job summaries` producono un markdown per portafoglio
  coerente con le notizie in DB (nessuna notizia duplicata).
- **ACC-04** — `job predict` produce una predizione per strumento; valori nel range
  denormalizzato del modello.
- **ACC-05** — `job recommend` produce pesi con `sum(|peso|) ≈ 1` e solo `n_long`
  long + `n_short` short.
- **ACC-06** — Le API esposte restituiscono solo dati in sola lettura e rispondono
  entro i tempi previsti (RN-03).
