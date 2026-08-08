# 09 — UI/UX Design System ("Cyber Light HUD")

Questo documento definisce le specifiche di design, la palette colori, la tipografia
e lo stile visivo della webapp Angular per HAYAI v2.

L'ispirazione visiva unisce l'estetica **cyber/futuristica di stampo HUD (Heads-Up Display)**
(ispirata a terminali di trading avanzati e interfacce sci-fi con bordi tecnici, angoli
tagliati e dettagli geometrici) ma declinata su un **Tema Chiaro (Light Theme)** pulito,
ad alto contrasto e professionale.

---

## 1. Concept Visivo: Cyber Light HUD

- **Sfondi puliti e luminosi** (`#f8fafc`, `#ffffff`) per una consultazione comoda e priva di affaticamento visivo durante la revisione mattutina o del martedì.
- **Accenti Cyber Neon (Acid Lime / Elettrico)**: Dettagli, bordi attivi e bottoni principali evidenziati con un verde acido brillante (`#65a30d` o `#84cc16`), richiamando l'energia dei terminali cyberpunk ma su sfondo chiaro.
- **Contenitori HUD e Angoli Tagliati**: Pannelli e card con bordi tecnici geometrici (`border-slate-300`, angoli smussati con `clip-path` o linee di mirino `+`), metadati monospaced (`Trade #`, coordinate fittizie, ID versione modello).
- **Tipografia Futurista**: Font geometriche e monospaced per dare l'effetto "command center quantitativo".

---

## 2. Palette Colori (Light Cyber Theme)

| Ruolo | Colore Hex | Descrizione |
|---|---|---|
| **Background Primario** | `#f8fafc` (Slate 50) | Sfondo generale della webapp |
| **Surface / Card** | `#ffffff` | Sfondo dei pannelli e delle schede tesi |
| **Bordi / Linee Tech** | `#cbd5e1` (Slate 300) | Bordi dei contenitori HUD |
| **Testo Principale** | `#0f172a` (Slate 900) | Testo ad alta leggibilità |
| **Testo Secondario** | `#64748b` (Slate 500) | Metadati, timestamp, etichette |
| **Cyber Accent (Primary)** | `#65a30d` (Lime 600) | Bottoni di azione, bordi attivi, badge long |
| **Cyber Accent (Hover)** | `#4d7c0f` (Lime 700) | Hover su bottoni |
| **Signal Long / Success** | `#16a34a` (Green 600) | Indicatori rialzisti, guadagni |
| **Signal Short / Warning** | `#dc2626` (Red 600) | Indicatori ribassisti, perdite |

---

## 3. Tipografia

- **Font per Intestazioni & Display**: **`Rajdhani`** o **`Orbitron`** (Google Fonts). Caratteri geometrici, larghi e marcatamente futuristici.
- **Font per Dati Tecnici & Metadati**: **`Share Tech Mono`** o **`JetBrains Mono`** (Google Fonts). Numeri, prezzi, coordinate e ID in stile terminale.
- **Font per Testo Normale (Body)**: **`Inter`** o **`Space Grotesk`**. Pulito e leggibile.

---

## 4. Componenti UI Chiave

### 4.1 Scheda Tesi di Investimento (Investment Thesis Card)
* Pannello bianco con bordo sottile e angolo superiore sinistro tagliato.
* Header con simbolo in font futuristica (`AAPL`), badge verde acido `[ LONG ]` e peso in evidenza (`15.4%`).
* Sezione metadati in font monospaced: `QUANT: +0.84` | `SENTIMENT: BULLISH`.
* Box tesi di investimento con bordo laterale verde acido e testo della rationale di DeepSeek in italiano.

### 4.2 Terminal HUD Header
* Barra superiore con titolo `HAYAI v2 // QUANT TERMINAL`, indicatore di connessione MariaDB, versione del modello attivo e timestamp dell'ultimo job batch notturno.

### 4.3 Bottoni Cyber
* Sfondo verde acido (`#65a30d`), testo scuro o bianco in grassetto monospaced, effetto glow sottile al passaggio del mouse e bordi squadrati.

---

## 5. Implementazione Tecnica (Tailwind CSS)
Per realizzare questo design in Angular, si utilizzerà **Tailwind CSS** personalizzando la configurazione (`tailwind.config.js`) con i font futuristici e la palette colori Cyber Light.
