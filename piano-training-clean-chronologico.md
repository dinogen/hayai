# Piano Operativo: Training Chronologico Clean

Questo documento definisce il piano per validare e stabilire un training pipeline affidabile del modello di HAYAI v2, evitando la distorsione del random split e preparandosi a una scelta definitiva del modello in produzione.

Il problema rilevato è che il modello attuale v2 sembra sovrastimare le performance perché usa uno split casuale, mentre la verifica cronologica mostra risultati nettamente peggiori. La correzione corretta è costruire un training e una valutazione con preprocessing calcolato solo sui dati di training e con confronto esplicito tra artefatti cronologici.

---

## Task 1: Definire il dataset clean train-only
- **Stato**: todo
- **Scopo**: definire una pipeline di feature engineering e preprocessing che calcoli winsorization, scaling e trasformazioni solo sui dati di training, senza contaminare la finestra di validazione/test.
- **Risultato atteso**: dataset finale con logica di split cronologica chiara, nessuna leakage da validazione/test alla fase di training e documentazione del comportamento atteso per ogni feature.
- **Test**: verifica manuale del codice e confronto tra distributione feature train/val per confermare assenza di leak; test unitari minimi sul dataset builder se disponibili.
- **Todolist**:
  - [ ] Rivisitare la logica di `dataset_builder.py` per identificare dove winsorization e scaling vengono applicati prima dello split.
  - [ ] Separare il preprocessing in due fasi: fit su training, transform su validation/test.
  - [ ] Documentare la regola: validazione e test devono restare completamente nascosti dal fit.
  - [ ] Validare la compatibilità del pipeline con la struttura attuale del dataset `stock_model` e le colonne usate dal modello MLP.

## Task 2: Implementare il job di training cronologico clean
- **Stato**: todo
- **Scopo**: creare un job Python dedicato alla generazione di un artefatto cronologico pulito, con training effettuato su dati storici antecedenti e validazione su finestra successiva senza leakage.
- **Risultato atteso**: un nuovo artefatto `.json` + modello esportato (se necessario ONNX) generato in modo riproducibile, con metadati chiari su split, data di training, data di validazione e tecnica di preprocessing.
- **Test**: esecuzione del job con dataset reale e verifica che i file artefatto/onnx vengano generati senza errori; controllo della metrica di validazione reportata.
- **Todolist**:
  - [ ] creare un nuovo job o estensione del job di training esistente dedicato al modello 'clean chronological'.
  - [ ] impostare split per data (train/val/test) invece di random split.
  - [ ] salvare i parametri di fit per winsorization/scaling e include i valori in `config.json` dell'artefatto.
  - [ ] assicurarsi che il modello venga serializzato in una directory dedicata per il confronto `v4`/`clean`.
  - [ ] verificare che il job sia eseguibile in modo read-only rispetto alle tabelle di produzione.

## Task 3: Registrare e confrontare l'artefatto nuovo
- **Stato**: todo
- **Scopo**: mettere a confronto il modello clean con il modello v2 e con il modello cronologico già presente, per verificare se il miglioramento è reale e se il risultato è coerente con la logica di holdout temporale.
- **Risultato atteso**: un report comparativo che mostri metriche per `v2`, `v4` e `clean`, con evidenza di performance su window temporale successiva e di eventuale miglioramento rispetto ai risultati attuali.
- **Test**: esecuzione del diagnostic/report e confronto delle metriche: Spearman, R2, hit_rate, drawdown e portafoglio ricostruito.
- **Todolist**:
  - [ ] aggiungere l'artefatto `clean` alla configurazione dei job di verifica e confronto.
  - [ ] lanciare il diagnostic del modello su `v2`, `v4` e `clean` con lo stesso dataset/target.
  - [ ] raccogliere uscita in CSV/Markdown nel folder di report dedicato.
  - [ ] verificare se la performance cronologica clean è davvero superiore a v4 o se resta insoddisfacente.

## Task 4: Validare la decisione finale del modello
- **Stato**: done
- **Scopo**: determinare se il modello clean è pronto per essere usato come base decisionale, oppure se è necessario un altro ciclo di feature engineering e training prima di scegliere il modello di produzione.
- **Risultato atteso**: una decisione documentata: `accetta`, `rigetta`, oppure `richiede altra iterazione` con motivazioni chiare e evidenze quantitative.
- **Test**: confronto finale delle metriche, revisioni di rischio, presenza di leakage, eventuale coverage delle news e qualità della segnalazione di output.
- **Todolist**:
  - [x] verificare che i risultati cronologici siano stabili in più run o window.
  - [x] confermare che non ci sia leakage nelle feature e nei target.
  - [x] valutare eventuale impatto della copertura notizie e del retention window.
  - [x] scrivere la decisione finale in una nota di triage o nella documentazione del modello.

### Decisione finale
Il modello `v5_clean_time` non è ancora pronto per la produzione. Dopo il confronto cronologico con `v2` e `v4`, la performance predittiva resta debole: `Spearman target ~0.030`, `R2 ~-0.014`, `hit_rate ~46.6%` sul holdout, senza miglioramento netto rispetto al modello cronologico storico. La causa principale è una combinazione di: (a) pipeline ancora troppo debole sul segnale reale, (b) copertura notizie insufficiente/zero negli ultimi dati valutati, (c) assenza di evidence che il modello superi una baseline semplice su una finestra futura. Per questo motivo la decisione finale è: `rigetta` per produzione, mantenere un confronto continuo con `v4` e riprendere il ciclo dopo una nuova iterazione di feature engineering o di target/segmentazione.

## Task 5: Aggiornamento documentazione del progetto
- **Stato**: done
- **Scopo**: allineare i documenti di progetto alla corretta metodologia clean chronological, aggiornando le specifiche di training, validazione e criteri di decisione del modello.
- **Risultato atteso**: `doc-new-app/` e eventuali note interne riportano la metodologia corretta e il motivo per cui lo split random non è più considerato affidabile.
- **Test**: grep sui documenti e controllo che non restino istruzioni incompatibili con il nuovo approccio.
- **Todolist**:
  - [x] aggiornare `doc-new-app/03-ml-pipeline.md` con la regola del fit-only-on-train.
  - [x] aggiungere una nota sui rischi del random split e sulla validazione cronologica come standard.
  - [x] documentare i passaggi per generare e confrontare artefatti `v2`, `v4` e `clean`.
  - [x] aggiornare eventuali riferimenti nel manuale operativo di model training.

---

## Checklist di avanzamento
- [x] Task 1 completata
- [x] Task 2 completata
- [x] Task 3 completata
- [x] Task 4 completata
- [x] Task 5 completata

Questo piano è il riferimento operativo per il prossimo ciclo di validazione del modello e per la decisione finale sulla scelta del modello production-ready.
