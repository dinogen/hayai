# 15 — Pipeline notizie e riassunti markdown

Questo documento descrive come la nuova applicazione **scarica le notizie
finanziarie**, le salva nel database e genera **riassunti in markdown** per
portafoglio. Le notizie arrivano da **yfinance** (senza chiave API).

## 1. Fonti notizie (yfinance)

yfinance espone notizie senza autenticazione:

- **`Ticker(symbol).news`** — lista di notizie relative a un singolo simbolo.
  Ogni elemento contiene campi tipici:
  - `id` (identificatore notizia),
  - `title`,
  - `publisher` (editore),
  - `link` (URL),
  - `providerPublishTime` (epoch),
  - `type` (es. `STORY`),
  - `relatedTickers` (lista ticker correlati).
- **`yf.Search(query, news_count=N)`** — ricerca con notizie associate alla query.

Scelte operative:

- Il job `news` usa `Ticker(symbol).news` per **ogni strumento attivo dei
  portafogli**, con limite di notizie per strumento (es. 10, configurabile).
- `yf.Search` può essere usato come fonte complementare per query di portafoglio
  (es. settore), ma non è obbligatorio.
- **Dedup globale** tramite `source_id` (UNIQUE in `news`).

## 2. Job `news` (acquisizione)

Passi:

1. Recupera la lista degli strumenti attivi nei portafogli (join
   `portfolio → portfolio_instrument → instrument`).
2. Per ogni strumento: `Ticker(symbol).news` (con retry/backoff per limiti).
3. Per ogni notizia:
   - normalizza `providerPublishTime` (epoch → DATETIME);
   - mappa i campi su `news` (source_id, instrument_id, title, publisher, link,
     published_at, summary);
   - **upsert** su `source_id` (ignora i duplicati).
4. Registra in `job_run`: numero strumenti processati, notizie nuove/già presenti,
   errori.

Campi mappati (schema `news`, doc `13 §2.7`):

| Campo yfinance | Colonna DB |
|---|---|
| `id` | `source_id` |
| `title` | `title` |
| `publisher` | `publisher` |
| `link` | `link` |
| `providerPublishTime` | `published_at` |
| `summary` (se presente) | `summary` |
| — | `instrument_id` (dal loop strumento) |

## 3. Job `summaries` (generazione markdown)

Per **ogni portafoglio attivo** e per la **data corrente** genera un riassunto in
markdown a partire dalle notizie in DB per i suoi strumenti.

### 3.1 Template del riassunto

```markdown
# Riepilogo notizie — <nome portafoglio> — <data>

Data generazione: <timestamp>
Strumenti con notizie: <n>
Notizie totali: <n>

## <Symbol> — <Nome strumento>
Fonte: yfinance · <n> notizie

- **<Title>**
  - Editore: <publisher>
  - Data: <published_at>
  - <URL: <link>>
  - Ticker correlati: <related>

---
```

Dettagli del template:

- Raggruppamento per **strumento** (ordinato per data di pubblicazione
  decrescente).
- Ogni notizia con titolo, editore, data, link e ticker correlati.
- Intestazione con portafoglio e data del riassunto.
- Se non ci sono notizie: sezione "Nessuna notizia per oggi" (o omissione,
  configurabile).

### 3.2 Salvataggio

- **DB**: upsert in `news_summary` su `(portfolio_id, summary_date)` con il
  contenuto markdown.
- **File**: export opzionale in `data/summaries/<portfolio>/<date>.md`
  (per consultazione diretta e versionamento file).
- Il contenuto markdown viene servito alla webapp come **testo** e renderizzato
  (RF-54).

## 4. Considerazioni di qualità

- **Dedup**: chiave `source_id`; notizie già presenti vengono ignorate senza
  duplicare (`INSERT ... ON DUPLICATE KEY UPDATE`).
- **Freschezza**: il job `news` è giornaliero; una notizia già in DB non viene
  re-inserita, ma il riassunto giornaliero può citare notizie dei giorni
  precedenti (finestra configurabile, default 1 giorno).
- **Limiti yfinance**: possibili risposte parziali; il job non deve fallire per
  uno strumento singolo (traccia in `job_run` con `status='partial'`).
- **Privacy/ToS**: yfinance è per uso personale; i riassunti non vanno
  commercializzati.

## 5. Requisiti soddisfatti

- RF-20 → §1-2 (download notizie da yfinance).
- RF-21 → §2 (dedup via `source_id`).
- RF-22 → §3 (riassunto markdown per portafoglio/data).
- RF-23 → §3.2 (salvataggio DB + export file).
- RF-24 → §4 + API (filtri per portafoglio/strumento/data).
