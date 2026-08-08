---
name: crea-piano
description: Crea un piano operativo strutturato per HAYAI v2. Usa quando l'utente chiede di creare un piano, un "piano operativo", "fai un piano", "prepara un piano" o quando serve pianificare un lavoro multi-task (training, ingestione, feature, deploy, ecc.). Il piano deve essere salvato in un file che inizia per "piano" e ogni task deve avere stato, scopo, risultato atteso e todolist.
---

# Crea Piano Operativo

Ogni volta che devi pianificare un lavoro multi-task per il progetto, crea un
documento piano seguendo il formato del modello di riferimento
`piano-training-modello.md` (nella root del progetto).

## Regole obbligatorie

1. **Nome file**: il piano va salvato in un file il cui nome inizia per
   `piano` (kebab-case, in inglese o italiano), es. `piano-training-modello.md`,
   `piano-ingestione-notizie.md`, `piano-deploy-raspberry.md`. Salvalo nella
   root del progetto se non indicato diversamente.
2. **Titolo**: un heading `#` con "Piano Operativo:" seguito da un titolo
   descrittivo.
3. **Task numerati**: sezioni `## Task N: <titolo>` in ordine crescente
   (`Task 1`, `Task 2`, `Task 3`, ...). Ogni task copre un solo argomento
   ben delimitato.
4. **Ogni task DEVE contenere** queste voci, nell'ordine:
   - `- **Stato**: todo` (o `in progress` / `done` quando aggiornato durante
     l'esecuzione)
   - `- **Scopo**: <descrizione breve di cosa si vuole ottenere>`
   - `- **Risultato atteso**: <descrizione verificabile del deliverable>`
   - `- **Test**: <se servono test e quali>`
   - `- **Todolist**:` seguita da checklist checkbox (`- [ ]`), una riga per
     azione concreta e verificabile.
5. Lingua: il documento del piano in **italiano** (come da convenzioni progetto).

## Struttura del file

```markdown
# Piano Operativo: <Titolo Descrittivo>

Questo documento definisce il piano per <obiettivo generale>.

---

## Task 1: <Titolo Task>
- **Stato**: todo
- **Scopo**: <cosa si vuole ottenere>
- **Risultato atteso**: <deliverable verificabile>
- **Todolist**:
  - [ ] <azione concreta 1>
  - [ ] <azione concreta 2>

## Task 2: <Titolo Task>
- **Stato**: todo
- **Scopo**: <cosa si vuole ottenere>
- **Risultato atteso**: <deliverable verificabile>
- **Todolist**:
  - [ ] <azione concreta 1>
```

## Cosa fare dopo aver creato il piano

- Se richiesto, esegui le task in ordine numerico, aggiornando `**Stato**` e
  spuntando le voci della todolist man mano che procedi.
- Consulta sempre `doc-new-app/` prima di scrivere codice, come da AGENTS.md.
