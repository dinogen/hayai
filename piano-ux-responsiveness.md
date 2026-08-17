# Piano Operativo: Refactoring UI/UX, Sidebar Responsive & Code Efficiency (`piano-ux-responsiveness.md`)

Questo documento definisce il piano per il refactoring completo dell'interfaccia frontend di HAYAI v2, integrando il menu laterale a scomparsa, la rimozione del background decorativo, l'ottimizzazione per smartphone e i miglioramenti di efficienza su HTML, CSS e JavaScript/TypeScript.

---

## Task 1: Pulizia Stili, Background e Rimozione Boilerplate Root
- **Stato**: todo
- **Scopo**: Rimuovere il background decorativo a circuiti stampati (`body::before` e `body::after`), impostare un grigio molto chiaro pulito (`#f8fafc`) e ripulire il file `app.html` eliminando il boilerplate predefinito di Angular.
- **Risultato atteso**: Sfondo uniforme e pulito, nessun elemento SVG di sfondo pesante, e radice dell'applicazione pulita pronta per ospitare navbar e router-outlet.
- **Todolist**:
  - [ ] Rimuovere le regole CSS di `body::before` e `body::after` in `styles.css` e confermare il colore `--bg-primary: #f8fafc`.
  - [ ] Svuotare `app.html` rimuovendo il template di benvenuto Angular e inserendo correttamente la nuova struttura (`<app-navbar />` + `<main>` con `<router-outlet />`).

## Task 2: Refactoring Navbar in Menu Laterale a Scomparsa (Sidebar / Drawer) & Responsive Mobile
- **Stato**: todo
- **Scopo**: Trasformare la navbar fissa superiore in un menu laterale a scomparsa (responsive drawer) con pulsante hamburger per smartphone e gestione responsive ottimizzata.
- **Risultato atteso**:
  - Su mobile: Barra superiore con logo, titolo e pulsante hamburger (`☰`), che apre/chiude un drawer laterale scorrevole con transizione fluida e overlay scuro.
  - Su desktop: Navbar laterale o header compatto con menu adattivo.
- **Todolist**:
  - [ ] Aggiornare `navbar.component.ts` implementando lo stato `isMenuOpen` (Signal) e il pulsante hamburger per dispositivi mobili.
  - [ ] Aggiungere stili CSS dedicati al drawer mobile (animazioni di apertura/chiusura, overlay, z-index).
  - [ ] Configurare la chiusura automatica del menu al click su qualsiasi link di navigazione.

## Task 3: Refactoring HTML & CSS (Rimozione Stili Inline & Semantizzazione)
- **Stato**: todo
- **Scopo**: Risolvere le inefficienze di HTML e CSS evidenziate nella review, convertendo gli stili inline ripetuti nei componenti (`DashboardComponent`, `RecommendationsComponent`) in classi CSS centralizzate e adottando tag semantici.
- **Risultato atteso**: Codice HTML pulito, uso esteso di classi riutilizzabili (`.hud-card`, `.btn-cyber`, ecc.) in `styles.css` al posto di `style="..."`, e utilizzo di tag semantici (`<section>`, `<header>`, `<article>`).
- **Todolist**:
  - [ ] Estrarre gli stili inline ripetuti in `DashboardComponent` e spostarli in classi riutilizzabili in `styles.css`.
  - [ ] Estrarre gli stili inline ripetuti in `RecommendationsComponent` (card tesi, metriche) e spostarli in classi CSS in `styles.css`.
  - [ ] Introdurre tag HTML semantici (`<section>`, `<article>`, `<header>`) al posto dei `div` generici dove appropriato.

## Task 4: Ottimizzazione TypeScript & Type Safety
- **Stato**: todo
- **Scopo**: Migliorare la manutenibilità e la tipizzazione del codice TypeScript sostituendo i tipi generici (`any[]`) con interfacce dedicate.
- **Risultato atteso**: Modelli TypeScript tipizzati per raccomandazioni, mercati e metriche di portafoglio.
- **Todolist**:
  - [ ] Definire interfacce TypeScript (es. `RecommendationItem`, `MarketStatus`, `PortfolioDetail`) in un file di modelli condiviso (es. `core/models/app.models.ts`).
  - [ ] Tipizzare le proprietà e i segnali in `DashboardComponent` e `RecommendationsComponent` rimuovendo l'uso di `any[]`.

## Task 5: Ottimizzazione Smartphone & Verifica Responsive
- **Stato**: todo
- **Scopo**: Garantire la perfetta compatibilità e usabilità su smartphone eliminando problemi di overflow e adattando griglie e tabelle.
- **Risultato atteso**: Interfaccia impeccabile e scorrevole su smartphone (<640px) e tablet.
- **Todolist**:
  - [ ] Verificare e regolare le media query e i breakpoint di griglie e card per viewport mobili.
  - [ ] Aggiungere wrapper `overflow-x-auto` a tutte le tabelle per evitare il taglio del layout su mobile.

## Task 6: Build e Verifica Finale
- **Stato**: todo
- **Scopo**: Compilare la webapp e verificare l'assenza di errori di build o regressioni.
- **Risultato atteso**: Build Angular completata con successo (`npm run build`).
- **Todolist**:
  - [ ] Eseguire `npm run build` in `hayai-new\web`.
  - [ ] Verificare che non vi siano errori di compilazione TypeScript o template.
