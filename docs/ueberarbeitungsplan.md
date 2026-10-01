# DEPOT — Überarbeitungsplan: bessere Zuordnung, schnellere Verarbeitung

Stand: 2026-10-02. Status: **Phase 0–4 umgesetzt** — Ergebnisse und Messwerte in den
Abschnitten 5a (Phase 1), 5b (Phase 2), 5c (Phase 3) und 5d (Phase 4). Auf dem Server
bestätigt ist davon noch nichts. Ergänzt [plan.md](plan.md)
(Architektur-Ist-Stand) um eine priorisierte Überarbeitung. Grundlage sind nicht Vermutungen,
sondern (a) der komplette Code, (b) die 31 echten Dateilog-Einträge aus dem Produktivbetrieb,
(c) die tatsächlich abgelegten PDFs und (d) Live-Messungen gegen den echten Ollama-Server
(mit synthetischem Dokumenttext, keine privaten Daten).

## 1. Kurzfassung

Drei Dinge bringen mit Abstand am meisten, in dieser Reihenfolge:

1. **Digital erzeugte PDFs werden heute falsch behandelt** (Befund B1): mehrseitige werden
   komplett gerastert und neu OCR't (ein 130-seitiges PDF wurde so zu 169 MB und brauchte
   ~10 Minuten), einseitige liefern dem LLM gar keinen Text. Das ist gleichzeitig das größte
   Tempo-, Qualitäts- und Speicherproblem — und mit einem Textebenen-Check vor dem OCR behebbar.
2. **Der Titel des Dokuments wird als Signal verschenkt** (B2, B3): Dateiname und
   PDF-Metadaten-Titel fließen nicht in die Ordner-Entscheidung ein, ein Datum im Dateinamen
   wird ignoriert, und bei fehlgeschlagenem OCR (Fotos) wird gar nicht erst klassifiziert,
   obwohl der Dateiname oft alles Nötige sagt. Künftig sieht die Ordner-Entscheidung immer
   alle drei Quellen: Titel-Signale, extrahierten Absender/Titel **und** den Inhalt.
3. **Die Ordner-Entscheidung kennt nur Ordnernamen, nicht deren Inhalt** (B4): ein Index über
   die bereits abgelegten Dateien ("wo liegen Dokumente dieses Absenders schon?") ersetzt den
   gierigen Ebene-für-Ebene-Abstieg durch Kandidatenliste + eine einzige Entscheidung. Das
   löst die bekannte offene Grenze (kein Absender-Ordner-Match) und spart 2–4 LLM-Aufrufe.

Dazu kommen billige Tempo-Gewinne beim LLM (Modell warm halten, Prompt-Reihenfolge) und bei
WebDAV (Ordnerliste vom lokalen Mount) sowie ein paar echte Robustheits-Bugs aus den Logs.

## 2. Ausgangslage in Zahlen

**Produktiv-Logs (31 Ereignisse, 03.–30.09.2026):**

| Kennzahl | Wert |
|---|---|
| `DATUM-UNSICHER` | 12 von 31 (davon 5 OCR-Fehlschläge) |
| `OCR-FEHLGESCHLAGEN` → `Unsortiert` ohne Klassifikation | 5 von 31 (alles Fotos/JPG mit sprechendem Dateinamen) |
| Gemeldete Konfidenz = 0.95 | 13 von 26 klassifizierten — trennt gut/schlecht kaum |
| Dauer pro Dokument (Abstand der Log-Zeitstempel im Batch) | Scanner-PDF 1 Seite: 18–26 s · digitale PDFs 2–8 Seiten: 25–59 s · 130 Seiten: ~10 min |
| Falscher Fehler-Eintrag durch Doppel-Verarbeitung | 1 (`Errno 2`, 1 s nach erfolgreicher Ablage derselben Datei) |

**Live-Messung Ollama** (GTX 1060, `qwen2.5:7b-instruct-q4_K_M`, Ollama 0.33.2, echte
DEPOT-Prompts, 3500 Zeichen synthetischer Text):

| Aufruf | Prompt-Token | Dauer |
|---|---|---|
| Modell laden (nach >5 min Leerlauf entladen, `/api/ps` war leer) | — | **11,4 s** |
| `extract_content` | 2895 | 7,5 s Prompt + 2,7 s Ausgabe ≈ 10 s |
| Ordner-Schritt, heutiges Prompt-Layout (Ebene zuerst, Text zuletzt) | ~2200 | je **6–7 s**, bei jedem Schritt neu |
| Ordner-Schritt, Text zuerst / Ebene zuletzt (KV-Cache greift) | ~2200 | 1. Schritt 6 s, danach je **1,0–1,6 s** |
| Kompakte Entscheidung nur aus Absender + Titel + Kandidaten | 544 | **1,7 s** |

Lokaler Pfad heute also ≈ 10 s + 3 × 6,3 s ≈ **29 s LLM-Zeit** bei Tiefe 3 (plus 11 s beim
ersten Dokument). Das Prompt passt in den Kontext (kein Abschneiden gemessen, gleiche
Token-Zahl mit `num_ctx=8192`), `num_ctx` ist aber nirgends festgelegt.

**Ordnerbaum:** 293 Ordner, 18 Top-Level, Tiefe bis 8, 555 PDFs, davon 173 Dateien bereits
im Schema `YYYY-MM-DD …`. In der produktiven `DEPOT Config.json` ist
`use_anthropic_classifier: true` — die Ordner-Wahl läuft aktuell also über Claude und sieht
nur Absender + Titel + Ordnerpfade.

## 3. Befunde

### Zuordnung / Qualität

- **B1 — Digital erzeugte PDFs: gerastert oder ohne Text** ([ocr.py:73-92](../depot/ocr.py)).
  `--skip-text` überspringt Seiten mit Textebene, schreibt in die Sidecar-Datei dann aber nur
  einen Platzhalter ("[OCR skipped on page(s) …]", 5 Wörter) statt des Textes. Folge:
  - ≥ 2 Seiten: 5 Wörter < `MIN_WORDS_PER_PAGE × Seiten` → Retry mit `--force-ocr` → das
    ganze PDF wird gerastert. **Belegt an den abgelegten Dateien:** alle mehrseitigen digitalen
    PDFs enthalten nur noch Seitenbilder + Tesseract-Textebene (`GlyphLessFont`), z.B. 7 Seiten
    = 4,3 MB, 130 Seiten = 169 MB (zweimal abgelegt). Perfekter Originaltext wird durch
    fehlerbehaftetes OCR ersetzt.
  - 1 Seite: 5 Wörter ist genau die Schwelle → gilt als "OCR ok", das LLM bekommt den
    Platzhalter als Dokumenttext. **Belegt:** einseitiges digitales PDF mit 169 Wörtern echter
    Textebene wurde in 8 s mit Titel = Dateiname, ohne Datum, abgelegt.
  (Mechanismus aus Code + dokumentiertem ocrmypdf-Verhalten abgeleitet; lokal nicht
  nachgestellt, da auf dem Entwicklungsrechner kein Tesseract installiert ist.)
- **B2 — Titel-Signale bleiben ungenutzt.**
  - OCR fehlgeschlagen → keine Klassifikation, direkt `Unsortiert`
    ([pipeline.py:242-250](../depot/pipeline.py)), auch wenn der Dateiname das Dokument klar
    benennt. Ein Datum im Dateinamen wird nicht erkannt → doppeltes Datumspräfix
    (`<heute> <Datum aus Name> … (Datum unsicher)`).
  - Dateinamen mit Datum (`… 28-09-2026 13-52-56.pdf`) landen trotzdem als `DATUM-UNSICHER`.
  - PDF-Metadaten (`Title`, `CreationDate`) sind bei den meisten digitalen PDFs vorhanden und
    werden nie gelesen. `CreationDate` stimmte in der Stichprobe mit dem echten Datum überein.
  - Der Cloud-Pfad bekommt den Dateinamen nie ([classifier.py:471-475](../depot/classifier.py)).
- **B3 — Kein Pfad sieht Titel UND Inhalt.** Lokaler Abstieg: OCR-Text + Dateiname, aber
  nicht den gerade extrahierten Titel/Absender ([classifier.py:203-224](../depot/classifier.py)).
  Cloud: Titel + Absender, aber nichts vom Inhalt. Real: ein Schreiben ohne erkannten Absender
  ging nur mit seinem Kurztitel in die Cloud und bekam einen neuen Top-Level-Ordner, während
  alle Schwester-Schreiben desselben Absenders im selben Batch richtig landeten.
- **B4 — Ordner sind für das Modell nur Namen.** Was in einem Ordner liegt, ist unbekannt;
  der Absender-Hinweis greift nur, wenn ein Ordner fast genauso heißt wie der Absender. Das
  ist die in plan.md dokumentierte offene Grenze.
- **B5 — Datum.** Modell darf Datumsangaben frei erzeugen (Ziffern-Vermischung real
  aufgetreten); ein Datum bis 60 Tage in der Zukunft gilt als plausibel
  ([models.py:15](../depot/models.py)) — real wurde so ein Termin im Dokument (in der Zukunft)
  statt des Ausstellungsdatums übernommen.
- **B6 — Absender uneinheitlich.** Rechtsformen ("GmbH", "AG") und Adressteile (PLZ, Ort)
  landen trotz Prompt-Regel im Absender → uneinheitliche Dateinamen, schwächerer Abgleich.
- **B7 — Konfidenz ist Selbstauskunft des Modells**, Minimum über alle Schritte. Die Hälfte
  aller Dokumente meldet 0.95; Ein-Optionen-Schritte melden 1.0.
- **B8 — Keine Messbarkeit.** Jede bisherige Verbesserung war eine Einzelfall-Korrektur; ob
  eine Änderung insgesamt hilft oder schadet, ist nicht prüfbar.

### Geschwindigkeit

- **S1 — Siehe B1**: unnötiges Raster-OCR digitaler PDFs ist der größte Einzelposten.
- **S2 — Zweiter OCR-Durchlauf bei Bildern ist wirkungslos** ([ocr.py:81-90](../depot/ocr.py)):
  ein Bild hat keine Textebene, `--force-ocr` liefert exakt dasselbe wie der erste Durchlauf.
  Jedes textarme Foto läuft doppelt.
- **S3 — Modell wird kalt** (11,4 s), kein `keep_alive`, kein Vorladen während des OCR.
- **S4 — Prompt-Layout verhindert Cache-Wiederverwendung**: der veränderliche Teil steht vorn,
  der konstante Dokumenttext hinten → jeder Schritt wertet ~2200 Token neu aus (Messung oben).
- **S5 — Ordnerliste per 293 sequenziellen PROPFINDs** alle 5 Minuten
  ([webdav.py:119-130](../depot/webdav.py)), obwohl der Container den kompletten Baum bereits
  read-only gemountet hat (`/nextcloud-data`). Dazu pro Ablage ein PROPFIND je Pfadsegment in
  `mkcol` plus ein Verzeichnis-Listing nur für die Kollisionsprüfung.
- **S6 — Streng seriell**: OCR (CPU) und LLM (GPU) überlappen nie, obwohl sie verschiedene
  Ressourcen nutzen.
- **S7 — Transienter Retry wiederholt das komplette OCR** (bis zu 5×).

### Robustheit (aus Logs und Code)

- **R1 — Doppel-Einreihung**: dieselbe Datei wird zweimal verarbeitet (z.B. `on_created` +
  `on_moved`); der zweite Lauf scheitert mit `Errno 2` und zählt als Fehlversuch.
- **R2 — "Fehler (1/3 Versuche)" wird nie erneut versucht**, erst beim nächsten
  Container-Start ([pipeline.py:162-184](../depot/pipeline.py)). Es gibt keinen periodischen
  Sweep; ein verpasstes Watcher-Ereignis bleibt ebenfalls liegen.
- **R3 — Inhaltsgleiche Dokumente** werden als `… (2).pdf` doppelt abgelegt.

## 4. Zielbild

```
Datei im Scan Eingang
  │
  ├─ 1. Signale (deterministisch, kein LLM)                         neu: depot/signals.py
  │     Dateiname → Scanner-Muster (SCN_/IMG_/…) oder Nutzertitel; Datum im Namen
  │     PDF-Metadaten → Title, CreationDate
  │     Text → vorhandene Textebene (pymupdf) ODER OCR; nie beides erzwingen
  │     Datumskandidaten → alle Datumsangaben aus Text + Name + Metadaten
  │
  ├─ 2. Extraktion (1 lokaler LLM-Aufruf)
  │     title, correspondent, issue_date (nur aus den Kandidaten wählbar), Stichworte/Dokumenttyp
  │     danach deterministisch: Absender normalisieren, gegen bekannte Absender angleichen
  │
  ├─ 3. Kandidaten (deterministisch)                                neu: depot/folder_index.py
  │     Index vom lokalen Mount: je Ordner Pfad + Dateinamen darin
  │     a) Absender-Historie: wo liegen Dateien dieses Absenders schon?
  │     b) Absender ≈ Ordnername (heutiger Fuzzy-Hinweis)
  │     c) Wort-Überlappung Titel/Stichworte ↔ Ordnerpfad + Dateinamen
  │     d) immer dabei: alle Top-Level-Ordner als Auffangnetz
  │
  ├─ 4. Entscheidung (1 LLM-Aufruf, lokal oder Cloud)
  │     sieht: Dateiname/PDF-Titel + Absender + Titel + Inhalt + ≤ ~12 Kandidaten
  │            (voller Pfad, je 2–3 Beispiel-Dateinamen)
  │     antwortet: Kandidaten-Nummer (Schema-Enum → erfundene Ordner unmöglich)
  │                oder "neuer Unterordner unter Kandidat N"
  │
  └─ 5. Konfidenz aus Signalen statt Selbstauskunft → Ablage / Unsortiert
```

## 5. Umsetzung in Phasen

Jede Phase ist einzeln deploybar. Reihenfolge nach Nutzen pro Aufwand.

### Phase 0 — Messbar machen (Voraussetzung für alles Weitere)

- **Zeiten pro Stufe** (OCR, Ordnerliste, jeder LLM-Aufruf inkl. Ollamas
  `load/prompt_eval/eval_duration`, Upload) in Container-Log und kompakt in die Dateilog-Zeile
  (`… | ocr=12.3s llm=9.8s dav=1.1s`).
- **Eval-Werkzeug** `tools/eval.py` (nicht Teil des Images): nimmt einen lokalen
  `Dokumente/`-Baum (der Nextcloud-Sync-Ordner) als Wahrheit — jede dort von Hand abgelegte PDF
  mit Textebene ist ein Testfall "richtiger Ordner bekannt". Leave-one-out (die Datei selbst
  wird aus dem Index entfernt), je einmal mit neutralem Dateinamen und mit Originalnamen.
  Kennzahlen: exakter Ordner · richtiger Top-Level · höchstens eine Ebene daneben ·
  Anteil `Unsortiert` · **falsch bei hoher Konfidenz** (der gefährliche Fall) · Sekunden pro
  Dokument. Ergebnisse bleiben lokal (gitignored) — das Repo ist öffentlich, keine echten
  Dokumentnamen/-inhalte einchecken.
- Feste Regressionsliste aus den realen Problemfällen (lokal, nicht im Repo).
- **Abnahme:** eine Baseline-Zahl für den heutigen Stand (lokaler Abstieg und Cloud-Pfad).

### Phase 1 — Schnelle, risikoarme Korrekturen

| # | Änderung | Dateien | behebt |
|---|---|---|---|
| 1.1 | Textebenen-Check vor OCR: hat jede Seite echten Text → ocrmypdf überspringen, Text per pymupdf, **Original unverändert hochladen**. Gemischte PDFs: `--skip-text`, Text danach per pymupdf aus dem Ergebnis statt aus der Sidecar-Datei. `--force-ocr` nur noch, wenn eine vorhandene Textebene nachweislich Müll ist. | `ocr.py` | B1, S1 |
| 1.2 | Kein zweiter OCR-Durchlauf für Bild-Eingaben / PDFs ohne Textebene. | `ocr.py` | S2 |
| 1.3 | Dateiname auswerten: Scanner-Muster erkennen, sonst als Nutzertitel behandeln; Datum im Namen parsen (`YYYY-MM-DD`, `DD-MM-YYYY`, `DD.MM.YYYY`; Scanner-Zeitstempel nur als Obergrenze "nicht später als"). PDF-`Title`/`CreationDate` lesen. | neu `signals.py`, `pipeline.py` | B2 |
| 1.4 | OCR fehlgeschlagen, aber sprechender Dateiname → trotzdem klassifizieren (Titel/Datum aus dem Namen), Tag `OCR-FEHLGESCHLAGEN` bleibt. Nur ohne verwertbaren Namen direkt nach `Unsortiert`. | `pipeline.py`, `classifier.py` | B2 |
| 1.5 | Datum absichern: `issue_date` muss einem im Text/Namen tatsächlich vorkommenden Datum entsprechen, sonst verwerfen. Zukunftspuffer von 60 auf wenige Tage. Fallback-Kette: Modell (validiert) → Dateiname → PDF-`CreationDate` (nur digitale PDFs) → heute + Marker. | `models.py`, `signals.py` | B5 |
| 1.6 | Ordner-Entscheidung bekommt zusätzlich Titel + Absender (lokal) bzw. Dateiname/PDF-Titel (Cloud, siehe Entscheidung E1). | `classifier.py` | B3 |
| 1.7 | Ollama: ein wiederverwendeter Client, `keep_alive` (z.B. 30 min), Vorlade-Aufruf sobald eine Datei in die Queue kommt (Modell lädt parallel zum OCR), `num_ctx` fest setzen (gleich für alle Aufrufe, sonst lädt Ollama neu). | `classifier.py`, `pipeline.py` | S3 |
| 1.8 | Ordnerliste vom lokalen Mount (`os.walk`, nur Verzeichnisse) statt PROPFIND-Kaskade; WebDAV bleibt Fallback. `mkcol` überspringt Segmente, die laut Liste existieren. | `pipeline.py`, `webdav.py`, `config.py` | S5 |
| 1.9 | In-Flight-Menge gegen Doppel-Einreihung; fehlt die Quelldatei beim Start der Verarbeitung → still überspringen statt Fehler zählen. | `watcher.py`, `pipeline.py` | R1 |

**Abnahme:** digitale PDFs bleiben byte-identisch und brauchen in der OCR-Stufe ~1 s statt
25–600 s; `DATUM-UNSICHER`-Quote auf der Regressionsliste deutlich unter den heutigen 39 %;
Eval-Kennzahlen nicht schlechter als Baseline; bestehende 133 Tests grün, neue Tests für 1.1–1.5.
Ergebnis siehe 5a (195 Tests grün).

### 5a. Stand nach Phase 0 + 1 (2026-09-30)

Umgesetzt: alles aus Phase 0 und 1.1–1.9, dazu E3 (Original statt PDF, wenn kein Text erkannt
wird) und E4 (Duplikate, vorgezogen aus 3.5). Abweichungen vom Plan:

- **1.7:** `keep_alive=30m`, Vorladen parallel zum OCR und festes `num_ctx=8192` (liegt mit
  4,99 GB weiter vollständig im VRAM) sind drin. Der wiederverwendete Client entfällt — eine
  neue Verbindung pro Aufruf kostet im LAN nichts Messbares.
- **Prompt-Reihenfolge (aus 2.4 vorgezogen) — mit einer wichtigen Korrektur:** "Dokumenttext
  zuerst, Ebene zuletzt" in EINER Nachricht ließ das Modell bei 3 von 4 echten Dokumenten auf
  der Wurzelebene "stay" antworten, die das alte Prompt richtig weiterleitete. Als zwei
  Gesprächs-Turns (Dokument → kurze Bestätigung → Ebene) bleibt das wiederverwendbare Präfix
  erhalten, ohne diesen Effekt. Aufgefallen ist das nur durch den Vorher/Nachher-Lauf von
  `tools/eval.py` — Prompt-Änderungen ab jetzt immer so gegenprüfen.
- **Eval-Werkzeug:** Leave-one-out ist noch nicht nötig (es gibt noch keinen Ordner-Index) und
  kommt mit Phase 2. `--depot-path` vergleicht zwei Code-Stände auf denselben Dokumenten.

**Vorher/Nachher, lokaler Pfad** — 30 zufällig gezogene, von Hand einsortierte PDFs aus dem
echten Baum (165 Kandidaten-Ordner), neutraler Dateiname `scan.pdf`, gleiche Dokumente:

| Kennzahl | vorher | nach Phase 1 |
|---|---|---|
| exakter Ordner | 6 / 30 (20 %) | **13 / 30 (43 %)** |
| höchstens eine Ebene daneben | 1 | 1 |
| richtiger Top-Level-Ordner | 13 | 15 |
| `Unsortiert` (unsicher) | 2 | 4 |
| falsch bei hoher Konfidenz | 21 / 30 (70 %) | **12 / 30 (40 %)** |
| Sekunden pro Dokument (nur LLM) | 14,8 | 14,3 |

Einordnung: deutlich besser, aber 30 Dokumente sind eine kleine Stichprobe, und ein Dokument
wurde schlechter (vorher exakt, jetzt `Unsortiert`). Der Maßstab ist streng — als richtig zählt
nur genau der Ordner, in dem das Dokument liegt, auch wenn er fünf Ebenen tief ist. Trotzdem:
**der lokale Pfad ist mit 40 % "falsch bei hoher Konfidenz" noch nicht gut genug**, und die
verbleibenden Fehler haben fast alle dieselbe Ursache — die erste Entscheidung auf der
Wurzelebene fällt nur anhand von Ordnernamen (typisch: ein Versicherungsdokument zu einem
Fahrzeug landet im allgemeinen Versicherungs-Ordner statt beim Fahrzeug, wo alle
gleichartigen Dokumente bereits liegen). Genau das adressiert Phase 2 mit dem Ordner-Index.
Die Zeit pro Dokument ist kaum gesunken, weil der Abstieg jetzt häufiger tiefer geht statt
früh stehen zu bleiben; der Präfix-Cache spart pro weiterem Schritt, die Extraktion (~10 s)
dominiert.

Nicht geprüft werden konnte der OCR-Pfad mit echtem Tesseract (auf dem Entwicklungsrechner
nicht installiert) — die Tests ersetzen den ocrmypdf-Aufruf. Der Fast-Path für digitale PDFs
ist an echten Dateien geprüft (7 Seiten: 0,02 s, Original unverändert). **Nach dem Deploy auf
dem Server gegenprüfen:** ein Scanner-PDF, ein digitales PDF, ein Foto; die Dateilog-Zeile
zeigt jetzt `ocr=… llm=… dav=…`.

### Phase 2 — Zuordnung neu aufbauen

- **2.1 Ordner-Index** (`folder_index.py`): vom Mount gelesen, mit der Ordnerliste gecacht; je
  Ordner die Dateinamen, daraus Absender-Häufigkeiten (Schema `Datum Absender - Titel`). Weil er
  aus dem Live-Baum kommt, lernt er automatisch aus jeder manuellen Korrektur: verschiebt der
  Nutzer eine Datei, zählt sie ab dem nächsten Lauf für den richtigen Ordner.
- **2.2 Absender normalisieren**: Rechtsformen und Adressteile deterministisch entfernen, dann
  per Fuzzy-Match an bereits verwendete Absender angleichen (ein Absender = eine Schreibweise).
- **2.3 Kandidaten + Einzelentscheidung** ersetzt `_walk_folder_tree`. Antwort als
  Kandidaten-Nummer per Schema-Enum; `UNGUELTIGE-ORDNERWAHL` entfällt konstruktionsbedingt.
  Jahres-Unterordner (`…/2025`, `…/2026`) werden deterministisch anhand `issue_date` gewählt,
  nicht vom Modell.
- **2.4 Ein gemeinsames Prompt-Präfix** für Extraktion und Entscheidung: gleiches System-Prompt,
  Dokumenttext zuerst, Aufgabe zuletzt. Die Entscheidung bekommt so den vollen Inhalt praktisch
  kostenlos (gemessen: 0,1 s statt 4,6 s Prompt-Auswertung) — Titel **und** Inhalt, wie
  gewünscht. Das System-Prompt dabei straffen (heute ~1450 Token). Achtung, siehe 5a: die
  Aufgabe muss ein eigener Gesprächs-Turn sein, und jede Variante wird per Eval gegengeprüft.
- **2.5 Konfidenz aus Signalen**: hoch, wenn Absender-Historie eindeutig und Modellwahl
  übereinstimmend; gedeckelt bei Widerspruch, neuem Ordner, leerem Absender oder nur
  Dateiname als Grundlage. Schwelle anhand der Eval-Daten kalibrieren statt fest 0.6.
- **2.6 Cloud-Pfad** nutzt dieselben Signale, soweit E1 sie erlaubt (keine Namen abgelegter
  Dateien). Kein Eskalations-Modus (E2): Maßstab für Phase 2 ist der lokale Pfad allein.
- **Abnahme:** Eval auf größerer Stichprobe (≥ 100 Dokumente, Leave-one-out) — exakter Ordner
  und "falsch bei hoher Konfidenz" klar besser als der Stand aus 5a (43 % / 40 %); der bekannte
  Fall "Absender ohne eigenen Ordner" landet lokal richtig; LLM-Zeit ≈ 10–12 s pro Dokument.
  Ergebnis siehe 5b.

### 5b. Stand nach Phase 2 (2026-09-30)

**So läuft die Ordner-Entscheidung jetzt** (lokaler Pfad; Details in [plan.md](plan.md)):
Extraktion → Vorauswahl ohne LLM aus Ordnernamen und den Namen der dort abgelegten Dateien
(`candidates.py`) → ein LLM-Aufruf wählt unter den Kandidaten (mit Beispieldateien) und den
Hauptordnern → Jahres-Unterordner folgen aus dem Datum → Konfidenz aus dem Beleg.

Umsetzung der Planpunkte und Abweichungen:

- **2.1 Ordner-Index** — umgesetzt (`folder_index.scan_local_tree`, `candidates.py`).
- **2.2 Absender normalisieren** — umgesetzt (Rechtsform/Adresse entfernen, an die bereits
  verwendete Schreibweise angleichen). Neue Dateinamen tragen den Absender ohne "GmbH"/"AG".
- **2.3 Kandidaten + Einzelentscheidung** — umgesetzt, mit zwei Änderungen aus der Messung:
  - Die Antwort ist der **Ordnerpfad**, nicht die Nummer in der Liste. Nach einer Nummer gefragt,
    hing die Wahl von der Reihenfolge der Liste ab: bei 6 echten Dokumenten ergab die umgedrehte
    Liste 6-mal einen anderen Ordner; mit ausgeschriebenem (per Schema auf die Liste
    beschränktem) Pfad nur 1-mal.
  - Der Ebene-für-Ebene-Abstieg bleibt als Rückfall, wenn kein Kandidat, sondern nur ein
    Hauptordner gewählt wird. Nach einem gewählten Kandidaten steigt das Modell NICHT weiter
    ab — im ersten Versuch führte genau das von einem richtig gewählten Ordner in einen
    unpassenden, sehr speziellen Unterordner.
- **2.4 Gemeinsames Prompt-Präfix** — für die Entscheidung umgesetzt (Folge-Turn des
  Extraktions-Gesprächs). Das Extraktions-Prompt selbst ist unverändert und nicht gekürzt:
  dieses Eval misst nur die Ordnerwahl, eine Verschlechterung bei Titel/Datum wäre unbemerkt
  geblieben.
- **2.5 Konfidenz aus Signalen** — umgesetzt, einfacher als geplant: 0.9 belegt / 0.7 neuer
  Unterordner in belegtem Ordner / 0.5 unbelegt. "Belegt" heißt: Beleg-Stärke ≥ 5 oder im
  gewählten Ordner liegen bereits Dateien dieses Absenders. Die Selbstauskunft des Modells
  zählt nicht mehr. Unbelegte Entscheidungen gehen mit `Vorschlag: <Ordner>` in der Logzeile
  nach `Unsortiert`; mit `CONFIDENCE_THRESHOLD=0.5` würden auch sie einsortiert.
- **2.6 Cloud-Pfad** — bewusst unverändert (nur Absender-Normalisierung und Datumsprüfung
  wirken auch dort). Er ist mit diesem Eval nicht gemessen: das würde Dokumenttitel an
  Anthropic senden und API-Kosten verursachen.

**Messung — 120 von Hand einsortierte PDFs aus dem echten Baum** (165 Kandidaten-Ordner,
neutraler Dateiname `scan.pdf`, Leave-one-out, dieselben Dokumente für beide Stände):

| Kennzahl | Phase 1 | Phase 2 |
|---|---|---|
| exakter Ordner | 47 (39 %) | **60 (50 %)** |
| eine Ebene daneben | 8 | 4 |
| `Unsortiert` | 7 (6 %) | 41 (34 %) |
| **falsch einsortiert** | **58 (48 %)** | **15 (12 %)** |

(Der ursprüngliche Stand vor Phase 1 wurde nur auf den ersten 30 dieser Dokumente gemessen:
6 exakt, 21 falsch einsortiert.)

Einordnung:

- Von den 79 Dokumenten, die Phase 2 einsortiert, liegen 65 exakt oder eine Ebene daneben.
  Von den 15 falsch einsortierten sind 9 Kopien von Gehaltsabrechnungen, die im Baum bewusst
  als Nachweis in einem themenfremden Ordner liegen — die "falsche" Ablage beim Arbeitgeber
  ist dort die naheliegende. Ohne diese 9 sind es 6 Fehler bei 70 Ablagen; 3 davon treffen
  einen Geschwisterordner im richtigen Zweig.
- Der Preis: ein Drittel geht nach `Unsortiert`. Dort wäre der Vorschlag in 16 von 41 Fällen
  exakt richtig gewesen — zu wenig, um blind abzulegen. Der Testsatz enthält dabei viel, was
  keine typische Post ist (Handbücher, Lebensläufe, englische Anleitungen, Rechnungen von
  Einmal-Händlern); für wiederkehrende Absender ist der Anteil kleiner.
- Die Vorauswahl allein: der richtige Ordner (Jahresordner dem Elternordner zugerechnet) steht
  bei 60 % an erster Stelle und bei 82 % unter den ersten acht.
- Zeit: LLM-Anteil im realistischen Lauf ohne Cache 11,6 s pro Dokument (8–14,5 s; 25 andere
  Dokumente), gegenüber 14,3 s nach Phase 1.

**Bekannte Grenzen nach Phase 2:**

- Rein lexikalisch: "Curriculum Vitae" findet den Ordner "Lebenslauf" nicht, ein englisches
  Handbuch nicht den Ordner "Manuals". Das wäre der Nutzen eines Embedding-Modells (Phase 4).
- Geschwisterordner, die sich nur durch einen Vorgang unterscheiden (mehrere
  Änderungs-Ordner derselben Versicherung, zwei Fahrzeuge desselben Typs), kann auch die
  Vorauswahl nicht sicher trennen.
- Ein Absender, der viele verschiedene Dinge liefert (Marktplatz-Rechnungen), zählt als
  "bekannt", sobald eine Rechnung von ihm irgendwo liegt — im Testsatz 1 Fehlablage dadurch.
- Ohne lokalen Mount (nur WebDAV) kennt die Vorauswahl nur Ordnernamen; dann ist deutlich
  weniger belegt und es geht mehr nach `Unsortiert`.
- Der eigene Name des Nutzers in Titel und Dateinamen erzeugt gelegentlich Zufallstreffer.

### 5c. Stand nach Phase 3 (2026-10-01)

Umsetzung der Planpunkte und Abweichungen (Details zum Ablauf in [plan.md](plan.md)):

- **3.1 Zwei Stufen** — umgesetzt. `MAX_CONCURRENT_JOBS` zählt jetzt nur die OCR-Threads;
  die LLM-Stufe ist immer genau ein Thread. Die Übergabe zwischen den Stufen ist ein
  `PreparedDocument` (Hash, Signale aus Dateiname/PDF, OCR-Ergebnis, die Schalter aus
  `DEPOT Config.json` zum Zeitpunkt des OCR). Wird eine Datei aus dem Eingang genommen,
  während sie auf die LLM-Stufe wartet, wird sie still übersprungen statt als Fehler
  gezählt.
- **3.2 OCR-Ergebnis zwischenspeichern** — umgesetzt (`ocr_cache.py`), mit einer
  Abweichung: Schlüssel ist der SHA-256 des Scans, nicht Pfad+Größe+mtime — er wird für
  den Duplikat-Check ohnehin berechnet und überlebt ein Umbenennen der Datei im Eingang.
  Der Eintrag (Text + ggf. erzeugtes PDF) liegt unter `/scratch/ocr-cache`, wird nach der
  Ablage gelöscht, sonst nach 7 Tagen. Ein Wiederholungsversuch nach transientem Fehler
  in der LLM-Stufe und ein Container-Neustart mitten im Batch kosten damit kein zweites
  OCR mehr.
- **3.3 OCR-Optionen nach Messung** — Werkzeug fertig (`tools/ocr_bench.py`: Varianten
  ohne `--clean`, ohne `--deskew`, ohne `--rotate-pages`, `--optimize 0`, `deu+eng`,
  `--jobs 1/4`; je Zeit, Größe, Wortzahl und Textübereinstimmung mit dem heutigen Stand),
  gemessen am 2026-10-02 im DEPOT-Image auf dem Entwicklungsrechner (16 Kerne) mit 8
  echten Dateien (7 Scanner-PDFs mit zusammen 13 Seiten, 1 Foto ohne Text), jeweils mit
  `--force-ocr`. Entscheidungsregel: eine Option bleibt nur, wenn sie Text bringt, den die
  Variante ohne sie nicht liefert.

  | Variante | Zeit gesamt | relativ | erkannte Wörter | Text gleich wie bisher |
  |---|---|---|---|---|
  | bisher (`--deskew --clean --rotate-pages`) | 122 s | 1,00 | 2468 | — |
  | ohne `--clean` | 82 s | 0,67 | 2468 | 6 von 7 identisch, 1× 97 % |
  | ohne `--deskew` | 76 s | 0,62 | 2495 | 90–100 % |
  | ohne `--rotate-pages` | 114 s | 0,93 | 2468 | identisch |
  | nur OCR | 52 s | 0,43 | 2495 | 90–100 % |
  | `--optimize 0` | 110 s | 0,90 | 2468 | identisch |
  | `deu+eng` | 123 s | 1,01 | 2473 | 91–100 % |
  | `--jobs 1` / `--jobs 4` | 151 s / 123 s | 1,24 / 1,00 | 2468 | identisch |

  Konsequenz: **`--clean` ist entfernt** (ein Drittel der OCR-Zeit ohne messbaren Nutzen).
  `--deskew` bleibt trotz ähnlicher Kosten: die Testscans kamen alle gerade aus dem
  Scanner, sein Nutzen liegt bei schief fotografierten Seiten, die im Testsatz fehlen —
  "Text anders" heißt hier nicht "Text schlechter", dafür fehlt eine Referenz.
  `--rotate-pages` bleibt (7 %, schützt vor kopfstehenden Seiten). `--jobs` und
  `--optimize` bleiben beim Standard. `deu+eng` kostet nichts, ändert aber den Text
  deutscher Dokumente um einige Prozent; über `OCR_LANGUAGE=deu+eng` einstellbar, nicht
  Standard. Wiederholbar mit `tools/ocr_bench.py` (auch im Container auf dem Server).
- **3.4 Periodischer Sweep** — umgesetzt (`SWEEP_INTERVAL_SECONDS`, Default 600, 0 = aus).
  Nimmt nur Dateien, die seit 60 s unverändert sind und nicht gerade vom Watcher
  entprellt werden. Dazu eine Queue ohne Doppeleinträge (`workqueue.py`): Event, Sweep
  und Retry-Timer melden dieselbe Datei sonst mehrfach, und jede Meldung wäre ein
  eigener Verarbeitungs- bzw. Fehlversuch. Nach 5 transienten Wiederholungen bleibt die
  Datei liegen; der nächste Sweep ist der nächste Versuch (vorher: bis zum Neustart
  nichts mehr).
- **3.5 Duplikate** — war bereits umgesetzt (E4).
- **3.6 Ablage schlanker** — umgesetzt: die Kollisionsprüfung liest die vorhandenen
  Dateinamen vom lokalen Mount, wenn der Zielordner dort liegt (kein Request mehr statt
  MKCOL-Prüfung + PROPFIND); der PUT läuft immer mit `If-None-Match: *` und überschreibt
  daher nie — wird der Name zwischen Prüfung und Upload vergeben, antwortet Nextcloud
  mit 412 und DEPOT nimmt den nächsten freien Namen.
- **Zusätzlich (Robustheit, aus der Fehleranalyse für 3.1):** bricht die Verbindung
  zwischen Upload und Löschen des Scans weg, legte der Wiederholungsversuch bisher eine
  zweite Kopie "(2)" ab. Jetzt wird der Ablageort VOR dem Löschen in der sqlite-DB
  vermerkt (`processed.source_deleted`); der nächste Versuch erkennt die erfolgte Ablage
  und entfernt nur noch den Scan (Logzeile "Ablage war bereits erfolgt"). Bestehende
  Datenbanken werden beim Start um die Spalte ergänzt.

**Messung:** Der Durchsatzgewinn von 3.1 ist mit dem Offline-Testsatz nicht messbar (kein
Tesseract); ein Test belegt nur den Mechanismus (OCR von Dokument 2 läuft, während die
Klassifikation von Dokument 1 blockiert). Erwartung aus dem Mechanismus: bei einem Batch von
Scanner-PDFs sinkt die Zeit pro Dokument von OCR + LLM auf die längere der beiden Stufen;
bei digitalen PDFs (OCR ≈ 1 s) ändert sich nichts. Die echten Zahlen liefern die
`ocr=`/`llm=`-Werte in den Dateilogs nach dem Deploy (die Zeitstempel-Abstände zwischen
den Logs geben den Durchsatz, die Stufenwerte die Auslastung).

### 5d. Stand nach Phase 4 (2026-10-02)

**Embedding-Modell für die Vorauswahl** (`EMBEDDING_MODEL`, Standard: aus; Details in
[plan.md](plan.md)). Das Modell läuft über denselben Ollama-Server, aber auf der CPU
(`num_gpu: 0`): gemessen ~0,1 s pro Text, Kaltstart 2–3 s, kein VRAM-Konflikt mit dem
Chat-Modell. Ein Text pro Ordner (Pfad + Titel der abgelegten Dateien), Vektoren in einem
sqlite-Cache unter `/scratch`; pro Dokument fällt im Betrieb ein Embedding-Aufruf an.

Gemessen auf denselben 120 Dokumenten wie in 5b:

| Variante | exakt | eine Ebene daneben | `Unsortiert` | falsch einsortiert |
|---|---|---|---|---|
| nur Wörter (Stand 5b) | 60 | 4 | 41 | 15 |
| Embedding nur in der Reihenfolge, `bge-m3` | 60 | 4 | 41 | 15 |
| Embedding nur in der Reihenfolge, `qwen3-embedding:0.6b` | 60 | 4 | 43 | 13 |
| **+ Übereinstimmung als Beleg, `bge-m3`** | **65** | 4 | **36** | 15 |
| + Übereinstimmung als Beleg, `qwen3-embedding:0.6b` | 66 | 4 | 37 | 13 |

Was die Messung gezeigt hat:

- **Die Vorauswahl wird besser, das Endergebnis dadurch allein nicht.** Mit `bge-m3` steht
  der richtige Ordner öfter vorn (Platz 1: 42 → 47, unter den ersten drei: 52 → 62 von
  120), aber die Wahl des Modells und vor allem die Beleg-Regel ändern sich dadurch nicht:
  was nicht belegt ist, geht weiter nach `Unsortiert`.
- **Die Höhe der Ähnlichkeit taugt nicht als Beleg** (im häufigsten Wertebereich war der
  ähnlichste Ordner nur in 69 % der Fälle der richtige).
- **Die Übereinstimmung zweier unabhängiger Signale taugt:** wo Wort-Vergleich und
  Embedding denselben Ordner an erste Stelle setzen, war er in 63 von 69 Fällen richtig
  (ohne die 9 bewusst themenfremd abgelegten Gehaltsabrechnungen). Das ist jetzt ein
  zweiter Weg zu "belegt": gewählter Ordner = bester nach Wörtern = bester fürs
  Embedding → Konfidenz 0.8, Tag `WORT-UND-BEDEUTUNG-EINIG`. Im Test: 5 bzw. 6 Dokumente
  mehr exakt einsortiert, keine zusätzliche Fehlablage.
- Zwei verworfene Varianten: Embeddings der einzelnen Dateititel statt eines Textes pro
  Ordner (schlechter als die Wort-Vorauswahl) und ein Instruktions-Präfix für die Anfrage
  (kein Gewinn).

Einordnung: der Gewinn ist echt, aber klein (5–6 von 120), und die Stichprobe für die neue
Regel ist entsprechend dünn. Deshalb bleibt `EMBEDDING_MODEL` standardmäßig leer; empfohlen
ist `qwen3-embedding:0.6b` (`ollama pull qwen3-embedding:0.6b`, 0,6 GB): im Test das
beste Ergebnis und zwei Fehlablagen weniger; `bge-m3` (1,2 GB) liegt knapp dahinter. Die bekannte Grenze aus 5b
("Curriculum Vitae" ↔ `Lebenslauf`) löst es nur halb: der Ordner kommt jetzt in die
Vorauswahl und wird vorgeschlagen, einsortiert wird aber weiterhin nur mit Beleg.

**Modellwechsel** (zweiter Punkt von Phase 4): nicht erneut gemessen. Mit
`tools/eval.py --model <name>` ist der Vergleich jetzt ein einzelner Lauf (~40 min mit
neuer Extraktion); der frühere Einzelfall-Vergleich mit `gemma2:9b` sprach nicht für einen
Wechsel.

### Phase 3 — Durchsatz und Robustheit

- **3.1 Zwei Stufen**: OCR-Worker (CPU) → LLM-Worker (GPU, genau 1, damit der Prompt-Cache
  hält). OCR von Dokument n+1 läuft, während Dokument n klassifiziert wird.
- **3.2 OCR-Ergebnis zwischenspeichern** (Schlüssel: Pfad + Größe + mtime) → transiente Retries
  und Neustarts wiederholen kein OCR.
- **3.3 OCR-Optionen nach Messung**, nicht nach Gefühl: `--clean` (unpaper), `--optimize`,
  `--jobs`, Sprache `deu+eng`. Jeweils Zeit und Texterkennung auf Beispielscans vergleichen.
- **3.4 Periodischer Sweep** (z.B. alle 10 min) holt liegengebliebene Dateien und echte
  Wiederholungsversuche nach; Quarantäne nach 3 Versuchen bleibt.
- **3.5 Duplikat-Erkennung** per Inhalts-Hash in der vorhandenen sqlite-DB — bereits umgesetzt
  (siehe E4).
- **3.6 Ablage schlanker**: Kollisionsprüfung über den Mount bzw. `If-None-Match: *` statt
  Verzeichnis-Listing.
- **Abnahme:** siehe 5c. Ergebnis der OCR-Messung (3.3) wird dort nachgetragen.

### Phase 4 — Optional, nur wenn die Eval-Zahlen es rechtfertigen

- Embedding-Modell über Ollama für die Kandidatensuche (semantisch statt Wort-Überlappung);
  VRAM-Konkurrenz mit dem Modell (4,99 GB bei `num_ctx=8192`) auf der 6-GB-Karte beachten.
  Laut 5b der naheliegendste nächste Schritt für die Zuordnung: er zielt auf die Fälle, die
  heute mangels Wort-Überlappung in `Unsortiert` landen.
- Modellwechsel — erstmals objektiv vergleichbar über `tools/eval.py` statt über Einzelfälle.

## 6. Erwartete Wirkung

| Fall | heute | nach Phase 1–2 |
|---|---|---|
| Digitales PDF, mehrseitig | 25–59 s, gerastert, vielfache Dateigröße | OCR-Stufe ~1 s, Original unverändert |
| Digitales PDF, 130 Seiten | ~10 min, 169 MB | Sekunden, Originalgröße |
| Digitales PDF, 1 Seite | LLM sieht keinen Text | voller Originaltext |
| Foto mit sprechendem Namen | doppeltes OCR, `Unsortiert` | einfaches OCR, Zuordnung über den Namen |
| Erstes Dokument eines Batches | +11 s Modell-Ladezeit | lädt parallel zum OCR |
| Batch von Scanner-PDFs (Phase 3) | OCR + LLM nacheinander | OCR von n+1 während LLM von n |
| Retry/Neustart nach transientem Fehler (Phase 3) | OCR erneut | OCR aus dem Cache |
| LLM-Zeit lokaler Pfad | ≈ 29 s | ≈ 10–12 s |
| Absender ohne eigenen Ordner | lokal falsch (bekannte Grenze) | über Absender-Historie/Ordnerinhalt |

Die Zeilen zu OCR-Dauern nach dem Umbau sind Schätzungen aus dem Mechanismus; die LLM-Zeiten
beruhen auf den Messungen in Abschnitt 2. Phase 0 liefert die echten Vorher/Nachher-Zahlen.

## 7. Entscheidungen (vom Nutzer getroffen, 2026-09-30)

- **E1 — Was darf an die Cloud gehen?** Zusätzlich zu Absender, Titel und Ordnerpfaden jetzt
  auch der vom Nutzer vergebene Dateiname, der PDF-Titel und lokal erzeugte Stichworte.
  Weiterhin nie: OCR-Volltext, Namen bereits abgelegter Dateien. Umgesetzt in Phase 1.
- **E2 — Keine Cloud-Eskalation.** Es kommt kein Automatismus hinzu, der unsichere Fälle an
  die Cloud schickt. Ziel von Phase 2 ist, dass der lokale Pfad allein ausreicht; der
  bestehende, von Hand gesetzte Schalter `use_anthropic_classifier` bleibt unverändert.
- **E3 — Fotos ohne erkennbaren Text** werden als Originalbild abgelegt, nicht als PDF.
  Umgesetzt in Phase 1 (gilt ebenso für PDFs, in denen kein Text erkannt wurde).
- **E4 — Duplikate** (byte-gleicher Scan, dessen Erstablage noch an ihrem Ort liegt) gehen mit
  Tag `DUPLIKAT` nach `Unsortiert` und tragen `(Duplikat)` im Dateinamen; kein erneutes
  OCR/LLM. Wurde die Erstablage inzwischen gelöscht oder verschoben, wird normal neu
  verarbeitet. Umgesetzt (vorgezogen aus Phase 3.5).

## 8. Bewusst nicht geplant

- Kein Vision-LLM, keine Metadaten-Datenbank/Such-UI, kein Review-Bereich — die Grundsätze aus
  [plan.md](plan.md) bleiben.
- Kein Umbau der Lösch-Garantie: `webdav.delete()` bleibt an genau einer Stelle, fest auf den
  Scan-Eingang verdrahtet.
- Die bereits gerastert abgelegten digitalen PDFs werden von DEPOT nicht nachträglich
  repariert; die Originale existieren nicht mehr im Scan-Eingang. Wer die Originale noch hat,
  kann sie nach Phase 1 einfach erneut einwerfen.
