# DEPOT — Automatisierte Dokumenten-Ablage-Pipeline für Nextcloud

## Context

Der Nutzer sammelt physische Post (Rechnungen, Behördenbriefe, Gehaltsabrechnungen,
Versicherungsunterlagen, Motorrad-Rechnungen, Bußgeldbescheide etc.) und scannt sie in
unregelmäßigen Abständen batchweise in einen `Scan Eingang`-Ordner in seiner
selbstgehosteten Nextcloud (TrueNAS Scale, Docker, WebDAV-Zugriff) — bewusst INNERHALB von
`Dokumente/` (`Dokumente/Scan Eingang`), damit alles ein zusammenhängender Baum bleibt;
DEPOT schließt den Scan-Eingang-Pfad strukturell von der Klassifikation aus (siehe unten),
sodass das gefahrlos möglich ist. Danach durchläuft er
für **jeden** Scan manuell: Inhalt lesen, Ausstellungsdatum ermitteln, Datei sinnvoll
umbenennen, den passenden (mehrere Ebenen tiefen) Unterordner in der bereits gut
gepflegten `Dokumente/`-Struktur suchen oder neu anlegen, Datei dort ablegen. Das ist bei
größeren Batches mühsam und zeitintensiv.

Ziel ist eine schlanke, DIY-taugliche Automatisierung (kein Paperless-ngx o.ä., da diese
Systeme eine eigene feste Ablagestruktur erzwingen statt in eine bestehende,
organisch gewachsene Struktur einzusortieren). Sensible Dokumente (Gesundheit, Gehalt)
dürfen das eigene Netz nicht verlassen — Klassifikation läuft daher zwingend über ein
lokal gehostetes LLM (Ollama).

**Wichtige Randbedingung, die die Architektur geprägt hat:** Die ursprünglich verfügbare
GPU auf dem TrueNAS-Server war eine GTX 960 (2–4 GB VRAM, alte Maxwell-Architektur) — zu
schwach für ein Vision-LLM, das Scans direkt liest. Deshalb: klassisches OCR (Tesseract)
läuft auf der CPU, ein kleines lokales Text-LLM (nicht Vision) übernimmt nur die
Klassifikation anhand des erkannten Texts. (Siehe [infrastructure-setup.md](infrastructure-setup.md)
für den späteren Verlauf der GPU-Frage inkl. Kartentausch auf eine GTX 1060.)

Im Gespräch mit dem Nutzer wurde die ursprünglich vorgesehene Review-Zwischenstufe
verworfen: Es gibt **keinen** Review-Bereich. Dateien werden vollautomatisch direkt in die
echte `Dokumente/`-Struktur einsortiert; als Sicherheitsnetz dient stattdessen eine
laufend aktualisierte Logdatei, die der Nutzer im Nachgang durchsehen kann, um
Fehlzuordnungen manuell in Nextcloud zu korrigieren.

## Bestätigte Entscheidungen (aus Rückfragen mit dem Nutzer)

- **Scan-Struktur:** 1 Datei = 1 Dokument (keine Trennung nötig), Formate gemischt
  (PDF, teils mehrseitig, sowie JPG/PNG/TIFF).
- **OCR:** Tesseract + Preprocessing (nicht Vision-LLM), CPU-basiert, deutsches
  Sprachpaket. Ein kleines Text-LLM (Ollama) übernimmt ausschließlich die
  Klassifikationsentscheidung.
- **Trigger:** vollautomatisch per Datei-Watcher auf `Scan Eingang`.
- **Umgebung:** Docker-Container auf dem TrueNAS-Server selbst, neben Nextcloud.
- **Ordnerstruktur-Abfrage:** live bei jedem Lauf, kurz gecacht — vom lokalen Mount, wenn er
  den Baum abdeckt, sonst per WebDAV (siehe Datenfluss).
- **Kein Review-Bereich:** direkte Einsortierung + Logdatei pro verarbeiteter Datei
  `DEPOT Dateilog DD-MM-YYYY HH-MM-SS.txt` in `Scan Eingang/Depot Config/` (siehe
  `CONFIG_SUBFOLDER`). Der Watcher ist nicht-rekursiv, sieht diesen Unterordner also
  ohnehin nie als Scan-Eingang; die Namens-basierte Ignorier-Logik bleibt zusätzlich als
  Sicherheitsnetz für Alt-Dateien aus der Zeit vor diesem Unterordner bestehen.
- **Dateiname als Signal:** ein vom Nutzer bereits vergebener Dateiname fließt zusätzlich
  zum OCR-Text in die Klassifikationsentscheidung ein.
- **Unsichere Fälle:** landen in einem Fallback-Ordner `Dokumente/Unsortiert`, deutlich im
  Log markiert.
- **Neue Ordner:** werden automatisch nach dem Namensmuster bestehender Ordner angelegt,
  aber im Log besonders hervorgehoben, damit der Nutzer sie kurz gegenprüfen kann.
- **Einsortierung optional abschaltbar (`file_into_dokumente` in `DEPOT Config.json`,
  Default an):** wenn aus, entfällt der komplette Ordner-Abstieg (spart die Ollama-Aufrufe
  dafür) — es werden nur Titel/Datum/Absender extrahiert, nichts landet unter `Dokumente/`.
- **Zusätzliche flache Kopie optional (`save_processed_copy` in `DEPOT Config.json`,
  Default aus):** legt jedes verarbeitete Dokument zusätzlich (oder bei abgeschalteter
  Einsortierung: ausschließlich) umbenannt+durchsuchbar flach unter `Scan Eingang/
  Depot Config/Processed/` ab. Sind beide Schalter aus, gewinnt intern `file_into_dokumente`
  (mit Warnung geloggt) — DEPOT würde sonst den Scan löschen, ohne das Ergebnis irgendwo
  abgelegt zu haben. Bewusst genau wie `excluded_folders` direkt in `DEPOT Config.json`
  steuerbar (nicht per Env-Var): wird bei jeder Datei frisch neu gelesen, eine Änderung in
  Nextcloud wirkt also sofort auf die nächste Datei, ohne Container-Neustart.
- **Cloud-Fallback für die Ordner-Entscheidung optional (`use_anthropic_classifier` in
  `DEPOT Config.json`, Default aus):** delegiert NUR die Ordner-Wahl an Anthropic (Claude),
  Titel/Datum/Absender-Extraktion bleibt immer lokal über Ollama. An Anthropic gehen
  `correspondent` + `title` + die vollständige Ordnerpfad-Liste, seit 2026-09-30 (bewusste
  Nutzerentscheidung) zusätzlich — soweit vorhanden — der vom Nutzer vergebene Dateiname
  (keine Scanner-Namen), der PDF-Metadaten-Titel und 3–6 lokal erzeugte allgemeine
  Stichworte (ohne Ziffern/Namen, im Code gefiltert). NIEMALS der OCR-Text. Grund: die lokalen ~7-9B-Modelle scheitern nachweislich
  genau dort, wo für eine sinnvolle Kategorie-Entscheidung Weltwissen/Urteilsvermögen ohne
  Einblick in Ordnerinhalte nötig ist (siehe Architektur-Diagramm unten,
  `CORRESPONDENT_FOLDER_MATCH_THRESHOLD`-Grenze). Ist `ANTHROPIC_API_KEY` nicht gesetzt
  oder schlägt der Aufruf fehl, landet das Dokument (mit dem bereits lokal ermittelten
  Titel) direkt in `FALLBACK_FOLDER` — kein Fallback auf den lokalen Ordner-Abstieg, kein
  Retry der ganzen Pipeline.

## Architektur / Datenfluss

```
Scan Eingang (Nextcloud-Datenverzeichnis, i.d.R. Dokumente/Scan Eingang, read-only
              Bind-Mount für schnellen Lesezugriff)
   │
   ▼
watcher.py (watchdog Observer, on_created/on_moved + Startup-Sweep bei Container-Start
            + periodischer Sweep, Default alle 10 min)
   │  - ignoriert Dateinamen, die "DEPOT Dateilog" enthalten
   │  - ignoriert nicht-whitelisted Dateiendungen (.pdf .jpg .jpeg .png .tif .tiff)
   │  - Debounce: wartet bis Dateigröße ~2s stabil ist (Scanner schreiben inkrementell)
   │  - nicht-rekursiv: `Scan Eingang/Depot Config/` (Logs, `DEPOT Config.json`,
   │    `Processed/`, `_Fehlerhaft/`) wird dadurch strukturell nie als Scan-Eingabe
   │    betrachtet, ganz ohne Namensfilter
   │  - der periodische Sweep nimmt nur Dateien, die seit ≥ 60 s unverändert sind, und
   │    holt so verpasste Events und aufgegebene Wiederholungsversuche nach
   ▼
workqueue.py: eine Queue, die jeden Pfad höchstens einmal enthält (Event, Sweep und
              Retry-Timer melden dieselbe Datei oft mehrfach)
   ▼
pipeline.py — zwei Stufen, damit CPU und GPU gleichzeitig arbeiten:
   Stufe 1 "OCR" (MAX_CONCURRENT_JOBS Threads, Default 1) und
   Stufe 2 "LLM" (genau EIN Thread: das Modell hält den Kontext eines Dokuments, ein
   zweites würde Ollamas Prompt-Cache verdrängen). Während Dokument n klassifiziert
   wird, läuft bereits das OCR von Dokument n+1.
   │
   ├─► Stufe 1: Duplikat-Check: SHA-256 des Scans gegen die sqlite-DB; liegt derselbe
   │           Inhalt noch am damaligen Ablageort → kein OCR/LLM, Ablage in Unsortiert
   │           als "<Name der Erstablage> (Duplikat).ext", Tag [DUPLIKAT]. Sonderfall:
   │           war es DEPOTs eigener vorheriger Versuch, dem nach dem Upload die
   │           Verbindung wegbrach, wird nur noch der Scan aus dem Eingang entfernt.
   │
   ├─► signals.py (ohne LLM): Dateiname → Nutzertitel + Datum oder Scanner-Name
   │           (nur Scan-Zeitpunkt); PDF-Metadaten → Titel, Erstelldatum; alle
   │           Datumsangaben im Text als Prüfmenge fürs Ausstellungsdatum
   │
   ├─► ocr_cache.py: OCR-Ergebnis (Text + erzeugtes PDF) liegt unter /scratch, bis der
   │           Scan abgelegt ist — ein Retry nach transientem Fehler oder ein
   │           Container-Neustart mitten im Batch wiederholt kein OCR. Schlüssel ist
   │           der SHA-256 des Scans (ohnehin für den Duplikat-Check berechnet).
   │
   ├─► ocr.py: PDF mit vollständiger eigener Textebene (digital erzeugt) → KEIN OCR,
   │           Text per pymupdf, Original wird unverändert abgelegt. Sonst:
   │           Bilder → img2pdf → ocrmypdf --language deu --deskew (kein --clean mehr)
   │           --rotate-pages --skip-text --sidecar. Hatte das PDF teilweise Text,
   │           wird der Text aus dem Ergebnis-PDF gelesen (die Sidecar-Datei enthält
   │           für übersprungene Seiten nur einen Platzhalter). Retry mit --force-ocr
   │           nur noch, wenn eine vorhandene Textebene offensichtlich Müll ist.
   │           Kein verwertbarer Text → OCR_FAILED, das ORIGINAL (z.B. das Foto) wird
   │           abgelegt; mit sprechendem Dateinamen trotzdem Klassifikation über den
   │           Namen ([NUR-DATEINAME]), sonst Fallback mit Konfidenz 0.
   │
   ├─► Stufe 2 (ab hier): Ordnerliste per os.walk vom lokalen read-only-Mount, wenn er den Baum abdeckt
   │           (15 s gecacht), sonst wie bisher per webdav.py: PROPFIND auf Dokumente/
   │           (Depth:1 rekursiv, da Nextcloud kein Depth:infinity erlaubt), 5 Min.
   │           gecacht (self-erstellte Ordner sofort im Cache ergänzt). Gefiltert um
   │           `excluded_folders` aus DEPOT Config.json UND strukturell IMMER um
   │           `SCAN_EINGANG_WEBDAV_PATH` (sonst könnte der Klassifikator ein Dokument
   │           in/unter den Scan-Eingang zurück-einsortieren, Endlosschleife mit dem
   │           Watcher) UND `FALLBACK_FOLDER`/Unsortiert (das ist die Konfidenz-
   │           Notbremse, kein normales Klassifikationsziel — real aufgetreten: das
   │           Modell wählte Unsortiert selbst, mit 0.95 gemeldeter Konfidenz und ganz
   │           ohne Tag, was die Funktion als sichtbares "muss geprüft werden"-Fach
   │           unterlief)
   │
   ├─► classifier.py — nur wenn file_into_dokumente aktiv ist (sonst nur Schritt 1):
   │     1. extract_content(): ein Ollama-Aufruf, NUR OCR-Text + Dateiname (+ PDF-Titel),
   │        ohne Ordnerkontext → {title, issue_date, correspondent, keywords,
   │        confidence}. `correspondent` ist PFLICHTFELD im JSON-Schema (nicht
   │        optional) — ein Live-Test zeigte, dass das kleine Modell ein optionales Feld
   │        praktisch immer mit null beantwortet, selbst mit expliziter Prompt-Anweisung,
   │        ein PFLICHT-Feld aber zuverlässig befüllt. Leerstring "" bleibt als "wirklich
   │        kein Absender erkennbar" gültig. Danach ohne LLM: Rechtsform/Adresse vom
   │        Absender entfernen und an die bereits verwendete Schreibweise angleichen
   │        (`naming.normalize_correspondent`).
   │     2. candidates.rank_candidates() (seit 2026-09-30, ohne LLM): Vorauswahl der
   │        Ordner, in denen bereits Ähnliches liegt — aus Ordnernamen UND den Namen
   │        der dort abgelegten Dateien (vom lokalen Mount gelesen). Verglichen werden
   │        Absender, Titel, Stichworte, Dateiname/PDF-Titel; zusätzlich zählt, ob ein
   │        Ordnername wörtlich im Dokumenttext vorkommt. Jahres-Unterordner ("2024")
   │        werden ihrem Elternordner zugerechnet. Jeder Kandidat trägt eine
   │        Beleg-Stärke und die Anzahl der Dateien desselben Absenders.
   │        Grund: vorher sah das Modell nur OrdnerNAMEN und musste auf der Wurzelebene
   │        raten, was in einem Ordner liegt — die Ursache der meisten echten
   │        Fehlablagen (ein Versicherungsschreiben zu einem Fahrzeug landete im
   │        allgemeinen Versicherungs-Ordner statt beim Fahrzeug, wo alle gleichartigen
   │        Dokumente lagen; ein Schreiben eines Absenders ohne eigenen Ordner landete
   │        in einer beliebigen Kategorie). Weil die Vorauswahl bei jedem Lauf aus dem
   │        echten Baum entsteht, lernt sie aus jeder Korrektur von Hand.
   │        Optional (`EMBEDDING_MODEL`, seit 2026-10-01): ein Embedding-Modell auf
   │        demselben Ollama (auf der CPU, `num_gpu: 0` — die Karte ist mit dem
   │        Chat-Modell voll) bewertet zusätzlich die BEDEUTUNG: ein Text pro Ordner
   │        (Pfad + Titel der abgelegten Dateien, `candidates.folder_text`) gegen einen
   │        Text fürs Dokument (Absender, Titel, Stichworte, 400 Zeichen Auszug).
   │        Die Ähnlichkeit fließt nur in die REIHENFOLGE der Kandidaten ein
   │        (`candidates.rank_candidates(semantic=…)`), Beleg-Stärke und
   │        Absender-Dateien bleiben wörtlich; stark belegte Ordner bleiben vorn.
   │        Vektoren werden unter /scratch gecacht (`embeddings.py`).
   │     3. _pick_folder(): EIN Ollama-Aufruf wählt unter den Kandidaten (je mit bis zu
   │        drei Beispieldateien) und den Hauptordnern. Er ist ein Folge-Turn des
   │        Extraktions-Gesprächs (gleiches Präfix → der Dokumenttext wird nicht erneut
   │        ausgewertet, und die Entscheidung sieht Text UND Titel/Absender). Die
   │        Antwort ist der ORDNERPFAD, per JSON-Schema auf die angebotenen Pfade
   │        beschränkt — erfundene Ordner sind damit ausgeschlossen. Bewusst kein
   │        Listen-Index: nach einer Nummer gefragt, hing die Wahl von der Reihenfolge
   │        der Liste ab (bei 6 echten Dokumenten mit umgedrehter Liste 6-mal ein anderer
   │        Ordner), mit ausgeschriebenem Pfad nur 1-mal.
   │     4. Danach nur noch: Jahres-Unterordner anhand des (geprüften) Datums betreten
   │        bzw. anlegen — ohne LLM. Wurde kein Kandidat, sondern nur ein Hauptordner
   │        gewählt, folgt von dort der frühere Ebene-für-Ebene-Abstieg
   │        (_walk_folder_tree: pro Ebene "descend"/"stay"/"new_folder", ungültige
   │        Wahlen werden per Fuzzy-Match korrigiert oder gekappt).
   │     5. Konfidenz aus dem Beleg statt aus der Selbstauskunft des Modells (die bei
   │        richtigen wie falschen Antworten 0.9+ meldete): 0.9, wenn der gewählte
   │        Ordner stark belegt ist oder dort bereits Dateien dieses Absenders liegen;
   │        0.8, wenn der gewählte Ordner sowohl nach Wörtern als auch für das
   │        Embedding-Modell der beste ist ([WORT-UND-BEDEUTUNG-EINIG] — zwei
   │        unabhängige Signale; gemessen in 63 von 69 Fällen richtig); 0.7 für
   │        einen neuen Unterordner in einem belegten Ordner; 0.5 ohne Beleg.
   │        0.5 liegt unter dem Standard-`CONFIDENCE_THRESHOLD` (0.6): das Dokument geht
   │        nach Unsortiert, der Vorschlag steht in der Logzeile ("Vorschlag: …"). Wird
   │        es von Hand einsortiert, findet das nächste gleichartige Dokument es dort.
   │        Wer auch unbelegte Vorschläge automatisch abgelegt haben will, setzt
   │        `CONFIDENCE_THRESHOLD=0.5`.
   │        Der Cloud-Pfad (`use_anthropic_classifier`) ersetzt die Schritte 2–5 durch
   │        einen Anthropic-Aufruf mit der ganzen Ordnerliste und behält dessen eigene
   │        Konfidenz; er ist von dieser Überarbeitung unberührt.
   │
   │        **Determinismus (2026-09-03):** beide Ollama-Aufrufe liefen mit
   │        `temperature=0.1` OHNE festen `seed` — ein Live-A/B-Test zeigte, dasselbe
   │        Dokument lieferte über 4 Wiederholungen 2 unterschiedliche Titel-
   │        Formulierungen, mit `temperature=0` + festem `seed` dagegen 4/4 mal exakt
   │        dasselbe Ergebnis. Beide Aufrufe nutzen jetzt `temperature=0, seed=42`
   │        (`classifier._OLLAMA_OPTIONS`) — es gibt keinen Vorteil durch kreative
   │        Variation bei einer Aufgabe, bei der dasselbe Dokument immer gleich
   │        einsortiert werden soll.
   │
   │        **Prompt-Aufbau (2026-09-30):** auch jeder Schritt des Ebenen-Abstiegs
   │        sieht den extrahierten Absender/Titel, den Dateinamen und den PDF-Titel.
   │        Der Dokumenttext steht als eigener erster Gesprächs-Turn VOR der (pro
   │        Schritt wechselnden) Ebene, damit Ollama das ausgewertete Präfix
   │        wiederverwendet. Wichtig: Text und Ebene in EINER Nachricht ("Text zuerst")
   │        ließ das Modell bei 3 von 4 echten Dokumenten auf der Wurzelebene "stay"
   │        antworten — erst die Zwei-Turn-Form entschied wieder wie das alte Prompt.
   │        Prompt-Änderungen hier nur noch mit `tools/eval.py` gegenprüfen.
   │        Beide Aufrufe: `num_ctx=8192` fest, `keep_alive=30m`, Modell wird beim
   │        Start jeder Verarbeitung parallel zum OCR vorgeladen.
   │
   ├─► signals.resolve_issue_date(): Modell-Datum zählt nur, wenn es im Dokument
   │           wirklich vorkommt und nicht in der Zukunft (bzw. nach dem
   │           Scan-Zeitstempel) liegt; sonst Datum aus dem Dateinamen
   │           ([DATUM-AUS-DATEINAME]), sonst — nur bei digital erzeugten PDFs — das
   │           PDF-Erstelldatum ([DATUM-AUS-PDF-METADATEN]), sonst [DATUM-UNSICHER]
   │
   ├─► naming.py: Titel sanitizen, Datum validieren, Dateiname
   │           "YYYY-MM-DD [Absender - ]Titel.ext" bauen, Kollisionen auflösen
   │           ("(2)", "(3)", …). Die vorhandenen Namen kommen vom lokalen Mount, wenn
   │           der Zielordner dort liegt (kein Request), sonst per PROPFIND; der PUT
   │           selbst läuft mit `If-None-Match: *` und überschreibt daher nie — wird
   │           der Name in der Zwischenzeit vergeben, kommt 412 und der nächste freie
   │           Name.
   │
   ├─► webdav.py: MKCOL (falls neuer Ordner) + PUT (neue durchsuchbare PDF hochladen,
   │           je nach Konfiguration nach Dokumente/... und/oder flach nach
   │           Scan Eingang/Depot Config/Processed/) + DELETE (Original löschen, NUR
   │           unter Scan-Eingang-Pfad — der komplette Code hat exakt eine Stelle, die
   │           webdav.delete() aufruft, und die ist hart auf den Scan-Eingang-Pfad
   │           fest verdrahtet; nirgendwo im Code wird je ein Ordner gelöscht) — alle
   │           Schreiboperationen laufen ausschließlich über WebDAV, NICHT über den
   │           Bind-Mount, damit Nextclouds interner File-Cache synchron bleibt
   │           (direkte Dateisystem-Schreibzugriffe erzeugen sonst unsichtbare
   │           "Ghost-Dateien" bis ein manueller `occ files:scan` läuft)
   │
   └─► depotlog.py: eigene Logdatei pro Verarbeitungs-Event unter
               Scan Eingang/Depot Config/DEPOT Dateilog DD-MM-YYYY HH-MM-SS.txt
               schreiben, inkl. Sondermarkierung für [OCR-FEHLGESCHLAGEN],
               [UNSORTIERT], [NEUER-ORDNER], [PROCESSED-KOPIE],
               [EINSORTIERUNG-DEAKTIVIERT], [NUR-DATEINAME], [DUPLIKAT] und die
               Datumsquelle; jede Zeile enthält die Dauer der Stufen
               (`ocr=…s llm=…s dav=…s`)
```

Die laufende Überarbeitung (Befunde aus dem Produktivbetrieb, Phasen, Messwerte) steht in
[ueberarbeitungsplan.md](ueberarbeitungsplan.md).

## Tech-Stack

- **Python 3.12** im Docker-Container.
- `watchdog` — Dateisystem-Events.
- `ocrmypdf` (kapselt tesseract, unpaper, ghostscript, qpdf) + `img2pdf` für lose
  Bilddateien → ein einziger Codepfad für alle Dateitypen.
- System-Pakete im Image: `tesseract-ocr`, `tesseract-ocr-deu`, `ghostscript`, `unpaper`,
  `qpdf`.
- `pymupdf` — schneller Check, ob ein PDF schon eine Textebene hat, sowie Seitenzählung.
- `httpx` + `xml.etree.ElementTree` — schlanker, selbstgeschriebener WebDAV-Client
  (PROPFIND/MKCOL/PUT/GET/DELETE/MOVE); bewusst keine vollwertige WebDAV-Library, passt
  zum Wunsch nach wenig Abhängigkeiten.
- `ollama` (offizieller Python-Client) — Chat-Aufruf mit JSON-Schema-Format; optional
  `embed()` mit einem Embedding-Modell (`EMBEDDING_MODEL`, z.B. `qwen3-embedding:0.6b`) für die
  Ordner-Vorauswahl.
- `anthropic` (offizieller Python-Client) — optionaler Cloud-Fallback für die
  Ordner-Entscheidung (`use_anthropic_classifier`), `messages.parse()` mit strukturierter
  Pydantic-Ausgabe. Nur diese eine Entscheidung, nie Titel/Datum-Extraktion und nie der
  OCR-Text selbst.
- `pydantic` — Schema-Validierung der LLM-Antwort.
- `pathvalidate` — Dateiname-Sanitizing (Umlaute bleiben erhalten, nur echte
  Sonderzeichen wie `/ \ : * ? " < > |` werden entfernt).
- stdlib `sqlite3` — kleine lokale Statusverfolgung (Fehlversuche pro Datei), um nach 3
  permanenten Fehlversuchen automatisch in einen Fehlerordner zu quarantänisieren
  (transiente Fehler wie Ollama/WebDAV nicht erreichbar zählen nicht mit).

Empfohlenes Modell: `qwen2.5:7b-instruct-q4_K_M` (starkes Deutsch). Läuft anfangs
CPU-only auf dem TrueNAS-Server; siehe [infrastructure-setup.md](infrastructure-setup.md)
für den aktuellen Stand zu GPU-Beschleunigung.

## Dateiname-Konvention

`YYYY-MM-DD [Absender - ]Titel.ext`, z.B. `2026-07-15 Stadtwerke München -
Stromrechnung Juli.pdf`, ohne erkennbaren Absender weiterhin schlicht
`2026-07-15 Stromrechnung Juli.pdf`. Fehlt ein erkennbares Datum, wird das
Verarbeitungsdatum verwendet, der Titel erhält den Zusatz "(Datum unsicher)" und der
Logeintrag wird mit `[DATUM-UNSICHER]` markiert.

**Titel-Qualität nachgeschärft (2026-09-03), anhand zweier realer schlechter Titel aus der
Produktion:** (1) ein persönlich adressierter Brief ohne offiziellen Formularnamen bekam
den generischen Titel "Bürgerbrief" statt eines Titels, der das tatsächliche Anliegen
nennt — Prompt-Regel ergänzt: bei einem freien Brief (Anrede "Sehr geehrte(r)...", kein
Formular) den Titel aus dem Inhalt/Thema ableiten, nie eine generische Textsorten-
Bezeichnung. Live verifiziert: derselbe Brief liefert jetzt "Schreiben über Änderung der
Steuerklasse" statt "Bürgerbrief". (2) ein TÜV-Prüfbericht bekam keinen Hinweis auf das
geprüfte Fahrzeug — Prompt-Regel ergänzt: bei einem Dokument zu einem konkreten, ggf.
mehrfach vorhandenen physischen Objekt (Fahrzeug, Gerät) eine im Text vorhandene eindeutige
Kennung (amtliches Kennzeichen, Seriennummer) mit in den Titel aufnehmen. Live verifiziert:
"Prüfbericht B-XY 1234" statt nur "Prüfbericht".

**Absender als eigenes Feld (statt Teil des freien Titels):** Recherche zu bestehenden
Lösungen (v.a. paperless-ngx, das Korrespondent/Dokumenttyp/Titel als getrennte Felder
modelliert und per Template zusammensetzt, sowie allgemeine Records-Management-Konventionen
für gescannte Geschäftspost: `Datum_Absender_Dokumenttyp[_Referenz]`) zeigt durchgehend,
dass ein Datum-zuerst-Präfix (bereits vorhanden) plus ein separates, kurzes
Korrespondenz-Feld die Konsistenz deutlich verbessert, gerade weil kleine LLMs bei einem
einzigen freien "Titel"-Feld stark variierende Formulierungen für inhaltlich gleiche
Dokumente liefern (Absender fließt sonst unstrukturiert und uneinheitlich mit ein). Ein
drittes, striktes `document_type`-Enum-Feld (wie bei paperless-ngx) wurde bewusst NICHT
übernommen: DEPOT hat keine Metadaten-Datenbank/Such-UI, die davon profitieren würde – die
vorhandene, handgepflegte Ordnerstruktur übernimmt diese Kategorisierung bereits
strukturell. Umsetzung:
- `classifier.py`/`models.py`: `ContentExtraction.correspondent` ist ein PFLICHTFELD im
  JSON-Schema (Leerstring "" statt `None` bedeutet "kein Absender erkennbar" — siehe
  Architektur-Diagramm oben für den Live-Test, der zeigte, dass genau das nötig war, um
  das kleine Modell zuverlässig zur Extraktion zu bewegen), per Prompt-Regel explizit
  NICHT mehr redundant im `title` enthalten. Auf `ClassificationOutcome` bleibt es
  weiterhin `str | None` (Leerstring wird dort zu `None` normalisiert).
- `naming.py`: `build_filename(..., correspondent=...)` stellt `"{Absender} - "` voran,
  wenn vorhanden; `MAX_FILENAME_LENGTH` (150 Zeichen) kappt das Ergebnis hart, als
  Sicherheitsnetz gegen ausufernde OCR-Titel bei tief verschachtelten Nextcloud-Pfaden.
- `candidates.py`: der extrahierte Absender ist das stärkste Signal der Ordner-Vorauswahl
  (Ordnername und Namen bereits abgelegter Dateien) — siehe Architektur-Diagramm oben. Der
  frühere Sonderweg "Absender ≈ Ordnername → Abstieg dort beginnen" ist darin aufgegangen.

## Repo-Struktur

```
DEPOT-Document-Engine-Pipeline-OCR-Tool/
  .github/
    workflows/
      docker-publish.yml  # baut+pusht ghcr.io/.../depot:latest bei Push auf master
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
  run.py
  depot/
    config.py       # Env-Loading, dataclass
    watcher.py       # watchdog + Startup-/periodischer Sweep + Debounce
    workqueue.py     # Queue ohne Doppeleinträge
    pipeline.py      # zwei Stufen (OCR / LLM+Upload), Fehlerbehandlung, Uploads
    ocr.py           # Textebenen-Check, img2pdf/ocrmypdf-Wrapper, Qualitätscheck
    ocr_cache.py     # OCR-Ergebnisse bis zur Ablage aufbewahren (/scratch)
    signals.py       # Dateiname/PDF-Metadaten/Datumsangaben auswerten (ohne LLM)
    folder_index.py  # Ordnerbaum samt Dateinamen vom lokalen Mount lesen
    webdav.py        # PROPFIND / MKCOL / PUT / GET / DELETE / MOVE, httpx-basiert
    classifier.py    # Content-Extraktion + Ordner-Entscheidung, Ollama-/Anthropic-Aufrufe
    candidates.py    # Ordner-Vorauswahl aus Ordner- und Dateinamen (ohne LLM), optional
                     # mit Embedding-Ähnlichkeit in der Reihenfolge
    embeddings.py    # Ollama-Embedding-Modell (CPU) mit sqlite-Vektor-Cache
    naming.py        # Sanitizing, Dateiname bauen, Kollisionen, Absender-Normalisierung
    depotlog.py      # Dateilog-TXT-Writer, ein File pro Verarbeitungs-Event
    scan_config.py   # DEPOT Config.json (excluded_folders) lesen/anwenden
    state.py         # sqlite: Fehlversuche + Hashes/Ablageort bereits abgelegter Scans
    models.py        # pydantic-Schemas (ContentExtraction, FolderPick, FolderStepDecision)
  tests/
    conftest.py       # Fake-Nextcloud-WebDAV-Server für Tests
    test_*.py
  tools/                # im Image enthalten, damit sie auch im Container laufen
    eval.py           # Klassifikations-Eval gegen einen handsortierten Baum
    ocr_bench.py      # ocrmypdf-Optionen auf Beispielscans messen (Zeit, Größe, Text)
  infra/
    ollama/
      docker-compose.yml  # Ollama-Stack für Dockge auf dem TrueNAS-Server
    depot/
      docker-compose.yml  # DEPOT-Stack für Dockge (image: ghcr.io/.../depot:latest)
  docs/
    plan.md                  # dieses Dokument
    infrastructure-setup.md  # TrueNAS/Ollama/GPU-Setup-Verlauf und Entscheidungen
```

**Kritische Dateien für die Umsetzung:** `depot/pipeline.py`, `depot/ocr.py`,
`depot/classifier.py`, `depot/webdav.py`, `docker-compose.yml`.

## Edge Cases

- **Korrupte/unlesbare Datei:** Exception in `ocr.py` abfangen, `[ERROR]` loggen,
  Fehlversuchszähler erhöhen, nach 3 Versuchen nach `Scan Eingang/Depot Config/
  _Fehlerhaft` quarantänisieren (bewusst NICHT unter `Dokumente/` — das ist DEPOTs eigener
  Quarantäne-Ordner, kein echtes Dokument, das der Klassifikator je als Ziel angeboten
  bekommen sollte).
- **Nicht unterstützter Dateityp:** Endungs-Whitelist, sonst `[SKIPPED-UNSUPPORTED]`
  loggen und unangetastet lassen.
- **Fast leerer OCR-Text:** erzwungene Konfidenz 0, Fallback nach `Unsortiert`,
  `[OCR-FEHLGESCHLAGEN]` im Log.
- **Ollama/Nextcloud nicht erreichbar/Timeout:** transienter Fehler (zählt nicht zum
  permanenten Fehlerlimit): bis zu 5 Wiederholungen im 30-s-Abstand, danach bleibt die
  Datei liegen und der nächste periodische Sweep versucht es erneut. Das OCR-Ergebnis
  bleibt dabei im Cache. Bricht die Verbindung zwischen Upload und Löschen des Scans
  weg, erkennt der nächste Versuch die bereits erfolgte Ablage und löscht nur noch.
- **WebDAV-Auth-Fehler:** Connectivity-Check beim Start, klarer Fehlschlag mit Log.
- **Ordner-Kollisionen/Fast-Duplikate:** Fuzzy-Match eines vorgeschlagenen neuen Ordners gegen die
  echten Kinder dieser Ebene — bei hoher Ähnlichkeit wird automatisch dorthin umgeleitet
  (`AUTO-REDIRECTED`/`AUTO-KORRIGIERT` im Log) statt einen Beinahe-Duplikat-Ordner
  anzulegen oder in `Unsortiert` zu landen.
- **Strukturell irrelevante Teilbäume** (z.B. ein riesiger Games/Amiibo-Ordner): über
  `excluded_folders` in der nutzereditierbaren `DEPOT Config.json` (in Scan Eingang/
  Depot Config) komplett von der Kandidatenliste ausschließen.
- **Scan-Eingang selbst als potenzielles Klassifikationsziel** (wenn er wie empfohlen
  unter `Dokumente/` liegt): strukturell und bedingungslos ausgeschlossen, unabhängig von
  `excluded_folders` — siehe Architektur-Diagramm oben.
- **Nicht-ASCII-Dateinamen:** NFC-Normalisierung vor jedem Vergleich/WebDAV-Pfad.
- **Große Batches:** zwei Stufen mit fester Thread-Zahl (OCR: `MAX_CONCURRENT_JOBS`,
  Default 1; LLM: immer 1), damit CPU (Tesseract) und GPU (Ollama) parallel, aber nie
  mehrfach belegt sind. Wird eine Datei aus dem Eingang genommen, während sie auf die
  LLM-Stufe wartet, wird sie still übersprungen.

## Verifikation / Testplan

1. `tests/fixtures/` mit ~10 repräsentativen Beispielen aufbauen, bevor der Watcher auf
   den echten `Scan-Eingang` zeigt: saubere PDF-Rechnung, schräg fotografierter JPG-Scan,
   verrauschter alter Behördenbrief, mehrseitiges PDF, nahezu leerer/fehlgeschlagener
   Scan, PDF mit bereits vorhandener Textebene, ein vorab umbenanntes Bild (testet den
   Dateiname-Signalpfad), ein Fall, der sinnvoll einen neuen Ordner auslösen sollte, ein
   Fall, der sinnvoll in `Unsortiert` landen sollte, ein ausgeschriebenes deutsches Datum.
2. Diese Beispiele durch `ocr.py` + `classifier.py` gegen eine echte lokale
   Ollama-Instanz laufen lassen, aber mit einer statischen `folder_tree.json`-Fixture
   (nicht live WebDAV) für schnelle, wiederholbare Durchläufe.
3. Harte Asserts für mechanische Korrektheit (nicht-leerer OCR-Text,
   Schema-Validierung, Dateiname-Sanitizing, Datumsparsing); manuell durchgesehene
   Diff-Tabelle für die naturgemäß unscharfe Klassifikationsqualität.
4. Erst danach den Watcher auf den echten `Scan-Eingang` ansetzen — zunächst mit
   `MAX_CONCURRENT_JOBS=1` und manueller Kontrolle der Dateilog-Einträge für die ersten
   ein bis zwei Batches.

Umgesetzt wurde bereits eine Offline-Testsuite (275 Tests) für alle Module, die ohne
echte Tesseract-/Ollama-/Nextcloud-Infrastruktur laufen (reine Logik, ein selbstgebauter
Fake-WebDAV-Server über `httpx.MockTransport`, gemockte Ollama-Aufrufe). Die in Schritt 1–2
beschriebenen Tests mit echten Beispiel-Scans stehen noch aus, sobald reale Dokumente zur
Verfügung stehen.
