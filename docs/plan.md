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
- **Ordnerstruktur-Abfrage:** live per WebDAV bei jedem Lauf (kein Cache).
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

## Architektur / Datenfluss

```
Scan Eingang (Nextcloud-Datenverzeichnis, i.d.R. Dokumente/Scan Eingang, read-only
              Bind-Mount für schnellen Lesezugriff)
   │
   ▼
watcher.py (watchdog Observer, on_created/on_moved + Startup-Sweep bei Container-Start)
   │  - ignoriert Dateinamen, die "DEPOT Dateilog" enthalten
   │  - ignoriert nicht-whitelisted Dateiendungen (.pdf .jpg .jpeg .png .tif .tiff)
   │  - Debounce: wartet bis Dateigröße ~2s stabil ist (Scanner schreiben inkrementell)
   │  - nicht-rekursiv: `Scan Eingang/Depot Config/` (Logs, `DEPOT Config.json`,
   │    `Processed/`, `_Fehlerhaft/`) wird dadurch strukturell nie als Scan-Eingabe
   │    betrachtet, ganz ohne Namensfilter
   ▼
pipeline.py (Worker-Loop, Concurrency konfigurierbar, Default 1)
   │
   ├─► ocr.py: Bilder → img2pdf → ocrmypdf --language deu --deskew --clean
   │           --rotate-pages --sidecar text.txt  (ein einheitlicher Codepfad für
   │           PDF und Bild). Qualitätscheck: bei zu wenig erkanntem Text Retry mit
   │           --force-ocr, danach ggf. OCR_FAILED → Fallback mit Konfidenz 0.
   │
   ├─► webdav.py: PROPFIND auf Dokumente/ (Depth:1 rekursiv, da Nextcloud kein
   │           Depth:infinity erlaubt) → flache Liste aller Unterordner, 5 Min. gecacht
   │           (self-erstellte Ordner sofort im Cache ergänzt), gefiltert um
   │           `excluded_folders` aus DEPOT Config.json UND strukturell IMMER um
   │           `SCAN_EINGANG_WEBDAV_PATH` (sonst könnte der Klassifikator ein Dokument
   │           in/unter den Scan-Eingang zurück-einsortieren, Endlosschleife mit dem
   │           Watcher) UND `FALLBACK_FOLDER`/Unsortiert (das ist die Konfidenz-
   │           Notbremse, kein normales Klassifikationsziel — real aufgetreten: das
   │           Modell wählte Unsortiert selbst, mit 0.95 gemeldeter Konfidenz und ganz
   │           ohne Tag, was die Funktion als sichtbares "muss geprüft werden"-Fach
   │           unterlief)
   │
   ├─► classifier.py — nur wenn file_into_dokumente aktiv ist (sonst nur Schritt 1),
   │   zwei getrennte Schritte statt einem Aufruf mit der ganzen Ordnerliste auf einmal
   │   (Grund: bei einer sehr großen/tiefen Struktur verliert ein kleines Modell sonst
   │   den Faden und wählt Unsinn — real aufgetreten):
   │     1. extract_content(): ein Ollama-Aufruf, NUR OCR-Text + Dateiname, ohne
   │        Ordnerkontext → {title, issue_date, correspondent, confidence}.
   │        `correspondent` ist PFLICHTFELD im JSON-Schema (nicht optional) — ein
   │        Live-Test zeigte, dass das kleine Modell ein optionales Feld praktisch immer
   │        mit null beantwortet, selbst mit expliziter Prompt-Anweisung, ein PFLICHT-
   │        Feld aber zuverlässig befüllt. Leerstring "" bleibt als "wirklich kein
   │        Absender erkennbar" gültig.
   │     2. _walk_folder_tree(): steigt Ebene für Ebene durch Dokumente/ ab. Pro
   │        Ebene ein Ollama-Aufruf mit nur den direkten Unterordnern DIESER Ebene
   │        (plus vollem Dokumenttext erneut) → "descend"/"stay"/"new_folder".
   │        Startet NICHT zwingend bei Dokumente/: matcht `correspondent` zuerst per
   │        Fuzzy-Vergleich (`closest_existing_leaf`, Schwelle 0.87) gegen JEDEN
   │        Ordner-Leaf-Namen im GESAMTEN Baum — bei einem Treffer beginnt der Abstieg
   │        direkt dort. Grund (real aufgetreten): ohne diesen Hinweis sieht das Modell
   │        auf der Wurzelebene nur 15 Ordnernamen ohne jeden Einblick, was darin
   │        eigentlich liegt, und wählte für eine Gehaltsabrechnung von "Bucher
   │        Grundstücksservice GmbH" die komplett falsche Kategorie "Finanzen" statt
   │        "Arbeit/Bucher Grundstücksservice" (der Ordner existierte bereits exakt so).
   │        Ab der falschen Wurzel-Wahl hatte jede weitere Ebene nur noch genau EINEN
   │        Unterordner zur Auswahl — "descend" war die einzig mögliche Antwort, und das
   │        Modell meldete an jeder dieser trivialen Ein-Optionen-Stufen konsequent
   │        Konfidenz 1.0, was die eine echte (falsche) Entscheidung ganz oben im Baum
   │        völlig verschleierte. Ungültige/halluzinierte Wahlen (erfundener
   │        "descend"-Zielname ohne Fuzzy-Match, oder leerer "new_folder"-Name) werden
   │        gegen die (kleine) Kandidatenliste dieser Ebene korrigiert oder sicher als
   │        "stay" behandelt — UND die für diesen Schritt gemeldete Konfidenz wird hart
   │        auf max. 0.2 gekappt (der Modellwert selbst ist in diesem Fall nicht
   │        vertrauenswürdig; vorher konnte ein halluzinierter Schritt mit z.B. 0.95
   │        gemeldeter Konfidenz das Dokument fälschlich sicher wirkend eine Ebene zu
   │        flach ablegen). Gesamt-Konfidenz = niedrigste Einzelkonfidenz über Inhalt +
   │        alle Schritte.
   │        **Bekannte, noch offene Grenze:** hat der Absender KEINE Entsprechung
   │        irgendwo im Baum (z.B. ein Finanzamt-Schreiben ohne existierenden
   │        "Finanzamt"-Ordner), greift der Fuzzy-Hint nicht und das Modell bleibt bei
   │        der ungelösten Wurzel-Entscheidung mit dem oben beschriebenen
   │        Ein-Optionen-Kaskaden-Problem — reale Fehlklassifikation, per Live-Test
   │        gegen die echte Ordnerstruktur bestätigt, noch nicht behoben.
   │
   ├─► naming.py: Titel sanitizen, Datum validieren, Dateiname
   │           "YYYY-MM-DD [Absender - ]Titel.ext" bauen, Kollisionen auflösen
   │           ("(2)", "(3)", …)
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
               [UNSORTIERT], [NEUER-ORDNER], [PROCESSED-KOPIE] und
               [EINSORTIERUNG-DEAKTIVIERT]-Fälle
```

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
- `ollama` (offizieller Python-Client) — Chat-Aufruf mit JSON-Schema-Format.
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
- `classifier.py`: der extrahierte Absender wird zusätzlich per Fuzzy-Match gegen
  existierende Ordner-Leaf-Namen im GESAMTEN Baum abgeglichen (`
  CORRESPONDENT_FOLDER_MATCH_THRESHOLD = 0.87`) und bestimmt bei einem Treffer den
  Startpunkt des Ordner-Abstiegs — siehe Architektur-Diagramm oben für den realen Fall,
  den das behebt, und die noch offene Grenze (kein Treffer = ungelöst).

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
    watcher.py       # watchdog + Startup-Sweep + Debounce + Queue
    pipeline.py      # Verarbeitung pro Datei
    ocr.py           # img2pdf/ocrmypdf-Wrapper, Qualitätscheck
    webdav.py        # PROPFIND / MKCOL / PUT / GET / DELETE / MOVE, httpx-basiert
    classifier.py    # Content-Extraktion + hierarchischer Ordner-Abstieg, Ollama-Aufrufe
    naming.py        # Sanitizing, Datumsparsing, Kollisionen, Fuzzy-Ordner-Match
    depotlog.py      # Dateilog-TXT-Writer, ein File pro Verarbeitungs-Event
    scan_config.py   # DEPOT Config.json (excluded_folders) lesen/anwenden
    state.py         # sqlite Fehlversuch-Tracker
    models.py        # pydantic-Schemas (ContentExtraction, FolderStepDecision)
  tests/
    conftest.py       # Fake-Nextcloud-WebDAV-Server für Tests
    test_*.py
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
- **Ollama nicht erreichbar/Timeout:** ~120s Timeout, ein Retry mit Backoff, danach
  transienter Fallback (zählt nicht zum permanenten Fehlerlimit, wird stattdessen
  automatisch requeued).
- **WebDAV-Auth-Fehler:** Connectivity-Check beim Start, klarer Fehlschlag mit Log.
- **Ordner-Kollisionen/Fast-Duplikate:** Fuzzy-Match auf jeder Abstiegs-Ebene gegen die
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
- **Große Batches:** eine `queue.Queue` + feste Worker-Zahl (Default 1), um CPU
  (Tesseract) und LLM (Ollama) auf bescheidener Hardware nicht zu überlasten.

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

Umgesetzt wurde bereits eine Offline-Testsuite (118 Tests) für alle Module, die ohne
echte Tesseract-/Ollama-/Nextcloud-Infrastruktur laufen (reine Logik, ein selbstgebauter
Fake-WebDAV-Server über `httpx.MockTransport`, gemockte Ollama-Aufrufe). Die in Schritt 1–2
beschriebenen Tests mit echten Beispiel-Scans stehen noch aus, sobald reale Dokumente zur
Verfügung stehen.
