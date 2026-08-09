# 04 — Pipeline Notizie & DeepSeek LLM Integration

Questo documento descrive come la nuova applicazione acquisisce le notizie
finanziarie da **yfinance**, le analizza tramite l'API di **DeepSeek** per estrarre
sentiment e tesi di investimento, e genera i riassunti in markdown per la webapp.

---

## 1. Acquisizione Notizie (`job news`)

Ogni notte, il batch esegue l'ingestione delle notizie:
1. Interroga la tabella `portfolio_instrument` per ottenere tutti i simboli attivi.
2. Per ogni simbolo, richiama `yfinance.Ticker(symbol).news`.
3. Filtra ed effettua l'**upsert** nella tabella `news` usando il campo nativo `id` di yfinance come `source_id` (garantendo l'idempotenza e prevenendo duplicati).

---

## 2. Analisi Semantica con DeepSeek API

Per ciascuna notizia fresca non ancora analizzata, il sistema invia una richiesta alle **API di DeepSeek** (modello `deepseek-chat` o equivalente raccomandato per task testuali e JSON mode).

### 2.1 Prompt Engineering Strutturato
Per evitare risposte ambigue o formati non parsabili, si forza il modello a restituire un **JSON valido**.

**Schema del Prompt inviato a DeepSeek:**
```text
Sei un analista finanziario quantitativo ed esperto di mercati.
Analizza la seguente notizia finanziaria relativa allo strumento finanziario {symbol} ({instrument_name}).

Titolo: {title}
Editore: {publisher}
Testo/Estratto: {summary}

Compito:
Valuta l'impatto di questa notizia sul prezzo a breve/medio termine dello strumento.
Restituisci UN UNICO oggetto JSON valido (senza blocchi markdown di contorno) con esattamente questa struttura:
{
  "sentiment": "bullish" o "neutral" o "bearish",
  "confidence": <valore float tra 0.0 e 1.0>,
  "catalyst": "<breve etichetta del catalizzatore, es. 'Earnings beat', 'Regulatory risk', 'Macro data', 'Product launch' o 'General'>",
  "rationale_it": "<Una spiegazione concisa e professionale in lingua italiana, di 2-3 frasi, che spieghi perché questa notizia influenza lo strumento e quale tesi di investimento supporta>"
}
```

### 2.2 Salvataggio in MariaDB
L'output JSON restituito da DeepSeek viene validato e salvato nella tabella `news_sentiment`, legata alla notizia in `news`.

---

## 3. Generazione del Segnale Ibrido (`portfolio_signal`)

Una volta analizzate le notizie del giorno per un dato strumento:
1. Si calcola un **Sentiment Score medio ponderato** per la giornata (pesato sulla `confidence` delle notizie recenti, es. ultime 24-48 ore).
2. Si converte il sentiment score in un `llm_sentiment_modifier` (range da `-0.20` a `+0.20`).
3. Si unisce al `quant_score` proveniente dal modello Keras per formare il `final_signal`.
4. Si consolida l'attributo `ai_rationale` unendo i punti chiave estratti da DeepSeek in un testo coerente, che costituirà la **motivazione** consultabile nella webapp.

---

## 4. Generazione Riassunti Markdown (`job summaries`)

Parallelamente all'analisi quantitativa, il batch genera un documento **Markdown giornaliero** per ciascun portafoglio, pensato per la lettura rapida dell'utente.

### Esempio di Markdown Generato (`news_summary`):
```markdown
# Riepilogo Notizie & Sentiment — Azionario Globale — 2026-08-08

### AAPL — Apple Inc. (Sentiment prevalente: 🟢 Bullish)
- **Apple announces breakthrough in custom silicon AI accelerators**
  - *Editore:* Reuters · *Data:* 2026-08-08 06:30
  - *Analisi IA:* L'annuncio rafforza il vantaggio competitivo nel segmento enterprise e AI edge, supportando una revisione al rialzo delle stime di margine.
  - [Leggi originale](https://finance.yahoo.com/news/...)

### QQQ — Invesco QQQ Trust (Sentiment prevalente: 🟡 Neutral)
- **Tech sector awaits upcoming inflation prints**
  - *Editore:* Bloomberg · *Data:* 2026-08-08 04:15
  - *Analisi IA:* Clima di attesa sui mercati azionari tecnologici in vista dei dati macroeconomici statunitensi; assenza di driver direzionali univoci.
  - [Leggi originale](https://finance.yahoo.com/news/...)
```

Questo file viene salvato sia nella colonna `markdown` della tabella `news_summary` sia esportato in una cartella locale per eventuale archivio.

---

## 5. Retention e Pulizia

Le notizie (e le relative `news_sentiment` in cascata) vengono conservate per **14 giorni**
per default. Il job batch `cleanup` (vedi `07-operativita-batch.md`) elimina le notizie
più vecchie del periodo di retention e rimuove anche i file cache parquet scaduti in `tmp/`
(`*_news.parquet` e `*_gnews.parquet`). Il periodo è configurabile con `--days`.
