# Infrastruktur-Setup: Ollama, Nextcloud und DEPOT auf TrueNAS SCALE

Dieses Dokument hält den tatsächlichen Setup-Verlauf und die Entscheidungen rund um die
Produktiv-Infrastruktur (lokales LLM, Nextcloud-Anbindung, DEPOT-Deployment) auf dem
TrueNAS-Server fest, damit sie nicht bei jeder Wartung neu recherchiert werden müssen.

## Server

- TrueNAS SCALE, Version 25.10.4 ("Goldeye"-Reihe).
- Erreichbar unter `https://10.69.69.230:444`.
- Nextcloud läuft selbstgehostet auf demselben Server via Docker.
- Dockge läuft ebenfalls auf dem Server zur Verwaltung eigener Compose-Stacks
  (unabhängig von TrueNAS' eingebautem "Apps"-System).

## Warum Dockge statt TrueNAS-Apps-Katalog

Die TrueNAS-eigene Ollama-Katalog-App bot keine nutzbare GPU-Konfiguration (nur eine
leere Überschrift ohne Auswahlmöglichkeit für NVIDIA-Karten). Dockge nutzt denselben
Docker-Daemon, erlaubt aber freie Compose-Dateien inkl. `deploy.resources.reservations`
für GPU-Zuweisung — und passt damit besser zum bestehenden `docker-compose.yml`-Ansatz
des Projekts.

## GPU-Verlauf

1. **Ausgangslage:** GTX 960 (2–4GB VRAM, Maxwell) im Server. Zu schwach für ein
   Vision-LLM, aber als Ziel für die Textklassifikation angedacht.
2. **TrueNAS-Apps-GPU-Konfiguration zeigte nichts an.** Grund: TrueNAS SCALEs
   eingebautes App-GPU-Passthrough deckt nur eine Checkbox "Passthrough available
   (non-NVIDIA) GPUs" ab — NVIDIA-Karten laufen über einen komplett anderen Mechanismus
   (System Extensions), nicht über diese Checkbox.
3. **Kartentausch auf eine GTX 1060.** Nach dem Tausch erkannte das System die neue
   Karte, aber ein Dockge-Deploy mit aktivierter `nvidia`-GPU-Reservierung schlug fehl:
   ```
   nvidia-container-cli: initialization error: nvml error: driver not loaded: unknown
   ```
4. **Ursache gefunden (offizielle TrueNAS-25.10-Versionshinweise):**
   > "TrueNAS 25.10 switches to open GPU kernel drivers supporting Turing and newer
   > (RTX/GTX 16-series+). Pascal, Maxwell, and Volta architectures are no longer
   > supported."

   Das heißt: Sowohl die alte GTX 960 (Maxwell) als auch die GTX 1060 (Pascal) fallen
   aus dem offiziell unterstützten Treiber-Pfad von TrueNAS 25.10 heraus — auch mit
   aktiviertem "Install NVIDIA Drivers"-Schalter im Apps-Pool, da dieser Schalter genau
   diesen (Pascal-inkompatiblen) offenen Kernel-Treiber installiert.
5. **Community-Treiber-Optionen recherchiert:**
   - [zzzhouuu/truenas-nvidia-drivers](https://github.com/zzzhouuu/truenas-nvidia-drivers) —
     vorgefertigte Legacy-Treiber-Builds (via `systemd-sysext`) für GTX 700/900/10-Serie.
     Nachteil: kein Build für die exakte Serverversion (25.10.4) verfügbar (nur
     25.10.5/25.10.6/26.0.0-BETA.3), Treiber-Blob stammt von einer privaten
     Drittanbieter-Domain.
   - [kaemis02/truenas-nvidia-extension](https://github.com/kaemis02/truenas-nvidia-extension) —
     **baut den Treiber selbst** aus offiziellen Quellen (offizielles TrueNAS-Update-Image
     + offizielles `truenas/scale-build`-Repo + offizieller NVIDIA-Treiber-Download),
     passend zu jeder TrueNAS-Version inkl. exakt 25.10.4. Laut README explizit getestet
     mit **"TrueNAS Scale 25.10.4 und NVIDIA GP106 [GeForce GTX 1060 6GB]"** — exakt unsere
     Kombination. Dafür entschieden, da vertrauenswürdiger als ein fertiger Blob unbekannter
     Herkunft.
6. **Zwischenzeitlich CPU-only betrieben**, während die GPU-Frage offen war (Begründung:
   Workload ist Batch-Klassifikation kurzer Texte, ca. 8–15 Tokens/Sekunde auf CPU sind
   grob 5–15 Sekunden pro Dokument — für unregelmäßige Scan-Batches praktikabel, auch wenn
   spürbar langsamer als gewünscht).
7. **Build-Versuche und Lessons Learned** (siehe auch Abschnitt "GPU-Treiber-Build" unten
   für den aktuellen Stand):
   - **Fehler #1:** Build zuerst direkt auf dem TrueNAS-Server selbst versucht. Falsch —
     das Projekt-README sagt explizit: *"The build is entirely offline from TrueNAS's
     perspective — you run it on any Linux machine with Docker, then upload the resulting
     file to your server."* Ein `--privileged` Docker-Build mit Chroot-/Squashfs-Manipulation
     hat auf dem produktiven Server (der auch Nextcloud betreibt) nichts verloren.
   - **Fehler #2 (Sackgasse):** Auf dem TrueNAS-Server schlug `apt install` innerhalb des
     Chroots mit `exit code 100` fehl, ohne dass die eigentliche Fehlermeldung sichtbar
     wurde (verschluckt vom `scale_build`-Basis-Codepfad). Eine manuelle Nachstellung des
     Chroot-Befehls zeigte `"Package management tools are disabled on TrueNAS appliances"`
     — TrueNAS' eingebauter Schutz gegen manuelle Paketverwaltung, eingebettet direkt im
     Root-Dateisystem-Image selbst. Diese manuelle Nachstellung war aber unvollständig
     (fehlende Vorbereitungsschritte, die das offizielle `run_in_chroot` normalerweise
     macht) und daher kein Beweis für die eigentliche Ursache — siehe Fehler #1, das
     Ganze hätte dort nie laufen sollen.
   - **Architektur-Falle:** Ein Linux-Laptop mit **Asahi Linux (Apple Silicon, ARM64)** ist
     ungeeignet als Build-Host — sowohl das Docker-Image als auch der fertige
     NVIDIA-x86_64-Treiber müssen zur Ziel-Architektur (x86_64 TrueNAS + x86_64 GTX 1060)
     passen. Cross-Build via QEMU-Emulation wäre theoretisch möglich, aber unnötiges
     Risiko bei einem ohnehin fragilen Build.
   - **Funktionierender Weg:** Windows-PC mit Docker Desktop + WSL2-Ubuntu-Distribution
     (`wsl --install -d Ubuntu`), da Docker Desktop selbst schon WSL2 als Unterbau nutzt.
     Docker Desktop → Settings → Resources → WSL Integration muss für die Ubuntu-Distro
     aktiviert sein. Zusätzliche Stolperfalle: der WSL-Linux-Nutzer muss in der
     `docker`-Gruppe sein (`sudo usermod -aG docker $USER`, danach Sitzung neu starten),
     sonst "permission denied" beim Docker-Socket. Der eigentliche Build (`generate.sh`)
     läuft NICHT unter Git Bash, da dieses kein `/proc/meminfo`/`nproc` bereitstellt, die
     das Skript zur Ressourcen-Erkennung braucht — nur in einer echten WSL2-Linux-Shell.
   - Fertige `nvidia.raw` wird per `scp` vom Build-Host auf den TrueNAS-Server übertragen.

## Aktuelles Ollama-Setup

Ollama läuft als eigener Dockge-Stack, siehe [`infra/ollama/docker-compose.yml`](../infra/ollama/docker-compose.yml).
Modelldaten liegen persistent unter `/mnt/tank/applications/ollama`.

Modell ziehen (einmalig, nach dem ersten Deploy):
```
docker exec -it ollama ollama pull qwen2.5:7b-instruct-q4_K_M
```

API erreichbar unter `http://10.69.69.230:11434`. In der DEPOT-Pipeline entspricht das
`OLLAMA_HOST` in der `.env` (siehe [.env.example](../.env.example)).

Bewusst **keine** expliziten CPU-Limits gesetzt: Linux/Docker verteilt CPU-Zeit per
CFS-Scheduler bereits fair zwischen Containern (gleiches Standardgewicht), Ollama ist
außerhalb aktiver Batches komplett inaktiv. Ein hartes Limit wäre vorbeugende Komplexität
ohne beobachtetes Problem — bei tatsächlich spürbarer Verlangsamung anderer Apps während
eines Batch-Laufs kann `deploy.resources.limits.cpus` in der Compose-Datei nachgerüstet
werden.

## GPU-Treiber-Build (Status: erledigt, GPU läuft)

1. Build mit [kaemis02/truenas-nvidia-extension](https://github.com/kaemis02/truenas-nvidia-extension)
   auf einer Windows-WSL2-Ubuntu-Umgebung (nicht auf dem TrueNAS-Server) erfolgreich
   durchgeführt — Ausgabe: `out-25.10.4/nvidia.raw`.
2. Datei per `scp` nach `/tmp/nvidia.raw` auf den TrueNAS-Server übertragen.
3. Installation auf dem Server selbst, laut Projekt-README (Option B — CLI):
   ```
   cp /usr/share/truenas/sysext-extensions/nvidia.raw /root/nvidia.raw.bak
   systemd-sysext unmerge
   zfs set readonly=off "$(zfs list -H -o name /usr)"
   cp /tmp/nvidia.raw /usr/share/truenas/sysext-extensions/nvidia.raw
   zfs set readonly=on "$(zfs list -H -o name /usr)"
   systemd-sysext merge
   systemctl restart docker
   systemd-sysext status
   nvidia-smi
   ```
4. **Bestätigt funktionierend:** `nvidia-smi` zeigt die GTX 1060 6GB korrekt (Treiber
   570.172.08, CUDA 12.8). Die GPU-Sektion im Ollama-Compose (siehe unten) ist aktiv.
5. Aufräumen nach den fehlgeschlagenen Host-seitigen Build-Versuchen erledigt:
   `rm -rf /mnt/tank/applications/nvidia-build` auf dem TrueNAS-Server ausgeführt.

**Nach jedem TrueNAS-Update, das den Kernel ändert:** `generate.sh` mit aktualisierter
`TRUENAS_VERSION`/`TRUENAS_TAG` erneut laufen lassen und neu installieren — die Extension
ist kernelversionsgebunden.

## Wechsel zu GPU (Ollama-Compose)

Im Ollama-Compose ist die GPU-Sektion jetzt aktiv:
```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
Die DEPOT-Pipeline selbst spricht ausschließlich mit der Ollama-HTTP-API und musste dafür
nicht verändert werden.

## GPU-Nutzung erneut verifiziert (2026-09-03)

Per `curl` gegen die Ollama-API (`/api/generate`, danach `/api/ps`) direkt gegenprüft, ob
die GTX 1060 tatsächlich noch für Inferenz genutzt wird (nicht nur `nvidia-smi` auf
Treiberebene): `size_vram` in der `/api/ps`-Antwort entsprach exakt der vollen Modellgröße
(4.75 GB) — das komplette Modell liegt im VRAM, kein CPU-Offload. Prompt-Verarbeitung lag
bei ~253 Tokens/Sekunde, deutlich über den zuvor dokumentierten ~8–15 Tokens/Sekunde
CPU-only-Werten. **Status: GPU-Beschleunigung ist aktiv und wird von Ollama tatsächlich
genutzt**, keine weiteren Schritte nötig.

## Modell-Experimente

- **Qwen2.5 7B Instruct (q4_K_M)** — Standardmodell, funktioniert mit dem hierarchischen
  Klassifikator (siehe `docs/plan.md`) grundsätzlich, aber nicht perfekt: bei einem
  Live-Test korrekt in `Dokumente/Behoerden/Wahlen` einsortiert, bei einem anderen
  Dokument einen nicht existierenden Unterordner erfunden statt den echten zu nehmen
  (Sicherheitsnetz griff, landete eine Ebene zu flach statt komplett falsch).
- **Llama 3.1 8B Instruct (q4_K_M)** ausprobiert, da Benchmarks es für strikte
  JSON-Schema-Einhaltung empfehlen — **Ergebnis: deutlich schlechter** für diese Aufgabe.
  Hat bei jedem getesteten Dokument sofort auf der Wurzelebene "stay" gewählt (landete
  direkt in `Dokumente/`, nie in einem Unterordner), mit auffällig identischer Confidence
  0.80 bei völlig unterschiedlichen Dokumenten — sieht nach einer Art
  Ausweich-/Standardantwort aus, wenn das Modell mit dem deutschsprachigen
  Mehrschritt-Prompt nicht klarkommt. **Wieder auf Qwen2.5 zurückgewechselt.**
- **Root-Cause-Analyse der schlechten Klassifikationsqualität (2026-09-03):** anhand
  echter Fehlklassifikationen aus den Produktions-Logs (drei Dokumente landeten
  fälschlich alle im selben Ordner `Finanzen/Vermögen/Scalable Capital/2026`) per
  Live-Repro-Skript gegen das echte Ollama + die echte Ordnerstruktur direkt
  reproduziert (nicht nur vermutet). Zwei konkrete Ursachen gefunden und behoben (siehe
  `docs/plan.md`, Architektur-Diagramm):
  1. `correspondent` kam bei klar erkennbarem Absender im Text (z.B. "Bucher
     Grundstücksservice GmbH" direkt im Briefkopf) trotz Prompt-Anweisung als `null`
     zurück. Ein isolierter Test bestätigte: als PFLICHTFELD im JSON-Schema (statt
     optional) extrahiert Qwen2.5 denselben Text zuverlässig korrekt. Grund vermutlich:
     ein optionales Feld im strukturierten Output ist für ein kleines Modell einfacher
     wegzulassen als zu befüllen, unabhängig von der Prompt-Anweisung.
  2. Die Wurzelebene (15 Top-Level-Ordner, reine Namensliste ohne jeden Einblick in den
     Ordnerinhalt) wählte für eine Gehaltsabrechnung "Finanzen" statt "Arbeit" —
     inhaltlich nicht absurd, aber falsch gegenüber der tatsächlichen
     Ablage-Konvention des Nutzers. Ab dort hatte jede weitere Ebene nur noch EINEN
     Unterordner zur Auswahl, sodass der komplette restliche Abstieg praktisch
     erzwungen war, aber trotzdem an jeder trivialen Stufe Konfidenz 1.0 meldete —
     die eine echte (falsche) Entscheidung ganz oben blieb dadurch unsichtbar.
     Fix: der extrahierte Absender wird jetzt zusätzlich per Fuzzy-Match gegen JEDEN
     Ordnernamen im gesamten Baum abgeglichen; bei einer Übereinstimmung (Schwelle 0.87)
     startet der Abstieg direkt dort statt an der Wurzel. Live verifiziert: die
     Gehaltsabrechnung landet jetzt korrekt unter `Arbeit/Bucher Grundstücksservice`
     statt `Finanzen/Vermögen/Scalable Capital`.
  3. Zusätzlich, unabhängig gefunden: `Unsortiert` war für den Klassifikator ein ganz
     normal wählbarer Ordner (kein struktureller Ausschluss) — das Modell wählte es in
     einem Fall selbst mit 0.95 Konfidenz, was den Zweck als sichtbares "muss geprüft
     werden"-Fach unterlief. Jetzt strukturell ausgeschlossen wie der Scan-Eingang-Pfad.
  - **War zu diesem Zeitpunkt weiterhin offen:** hat der Absender KEINE Entsprechung
    irgendwo im vorhandenen Baum (z.B. ein Finanzamt-Schreiben ohne existierenden
    "Finanzamt"-Ordner), greift der neue Fuzzy-Hint nicht, und die Wurzelebene blieb die
    ungelöste Schwachstelle — dieses Dokument landete im Live-Test weiterhin fälschlich
    unter `Finanzen/Vermögen/Scalable Capital`. Siehe unten, wie das gelöst wurde.

- **Gemma2 9B live getestet (2026-09-03), anhand des MÖVE-Benchmarks der Bundesdruckerei**
  ([arxiv.org/pdf/2606.13111](https://arxiv.org/pdf/2606.13111), ein Benchmark speziell für
  LLMs auf deutschen Verwaltungs-/Behördendokumenten): Gemma2 9B liegt dort bei "Topic
  Extraction" (die zu unserer Ordner-Klassifikation nächstverwandte Aufgabe) auf **Platz 1
  von 39 Modellen** (Score 0.671), vor Gemma3 27B, GPT-4o, Mistral Small 3.1 und Llama 3.3
  70B; bei German QA auf Platz 5 von 39. Qwen taucht in dieser Top-10 gar nicht auf. Live
  gegen die drei realen Testfälle aus dieser Session verglichen (`ollama pull gemma2:9b`,
  ~5.4GB, passt auf die GTX 1060) — **Ergebnis gemischt, kein klarer Sieger:** beim
  Bucher-GS-Fall und beim Finanzamt-Fall identisch zu Qwen2.5 (beim Finanzamt-Fall auch
  identisch falsch — der Fuzzy-Hint griff bei keinem der beiden, da keine Absender-
  Entsprechung existiert), beim TÜV-Prüfbericht-Fall aber sogar SCHLECHTER: landete mit
  0.95 Konfidenz unter `Arbeit/A Plus Transport/Arbeitgeber` (ein früherer Arbeitgeber,
  hat nichts mit einer Fahrzeugprüfung zu tun) statt wie Qwen2.5 im plausibleren
  `Dokumente/Zertifikate`. Lehre: allgemeine Sprachbenchmarks übertragen sich nicht
  zuverlässig auf DEPOTs konkrete Aufgabe (mehrstufige Navigation durch eine
  personalisierte, dem Modell unbekannte Ordnerstruktur) — das ist eine andere Fähigkeit
  als reine Themenextraktion aus einem einzelnen Text. Modell bleibt auf dem Server
  installiert, aber NICHT als Standard umgestellt.

## Cloud-Klassifikation via Anthropic (Claude) für den Ordner-Fall ohne Absender-Match (2026-09-03)

Der oben beschriebene, mit lokalen Modellen ungelöste Fall (kein Absender-Ordner-Match,
z.B. das Finanzamt-Schreiben) wurde live mit Claude Haiku 4.5 getestet — Ergebnis:
korrekt `Dokumente/Finanzen` (0.85–0.95 Konfidenz, ehrlich statt geschönt bei 1.0)
statt der falschen `Finanzen/Vermögen/Scalable Capital`-Kaskade. Auf Wunsch des Nutzers
umgesetzt als optionaler Schalter `use_anthropic_classifier` in `DEPOT Config.json` (siehe
`docs/plan.md` und `.env.example`) — delegiert NUR die Ordner-Entscheidung an die Cloud,
Titel/Datum/Absender bleiben immer lokal. An Anthropic geht bewusst NUR `correspondent` +
`title` + die Ordnerpfad-Liste, live verifiziert dass das ausreicht (Finanzamt-Fall sogar
mit höherer Konfidenz ohne Volltext als mit).

**API-Key-Setup (Lessons Learned):**
- Ein **Workspace-scoped API-Key** aus der Console (console.anthropic.com → Settings →
  API Keys, innerhalb eines konkreten Workspace angelegt), NICHT ein organisationsweiter
  "identity-linked" Key — letzterer verlangt einen zusätzlichen `anthropic-workspace-id`
  Header bei jedem Call, den ein einfacher `Anthropic(api_key=...)`-Client nicht mitgibt
  (Fehler: `anthropic-workspace-id is required when authenticating with an
  identity-linked API key`). Mit einem workspace-scoped Key trat das nicht auf.
- Abrechnung läuft komplett getrennt von einem claude.ai Pro/Max/Team-Abo — eigenes
  Pay-as-you-go-Billing in der Console, unabhängig vom Chat-Abo.
- `client.messages.parse(..., output_format=PydanticModel)` (structured outputs) nimmt in
  der installierten SDK-Version (`anthropic==1.3.0`) KEIN `temperature`/`top_p`/`top_k`
  entgegen — aktuelle Claude-Modelle (Opus 5, Sonnet 5, Fable-Serie) haben diese
  Sampling-Parameter aus der API entfernt (`effort` ersetzt sie für Denktiefe, steuert
  aber nicht die Ausgabe-Varianz auf dieselbe Art). Anders als beim Ollama-Fix
  (`temperature=0, seed=42`) gibt es hier also keinen Determinismus-Hebel — beobachtete
  Varianz blieb in Live-Tests aber innerhalb sinnvoller Optionen (z.B. "Finanzen" vs. das
  spezifischere "Finanzen/Steuern"), nie eine falsche Kategorie.
- Kosten bei DEPOTs Nutzungsvolumen (gelegentliche Scan-Batches, winziger Payload:
  Absender+Titel+Ordnerliste) vernachlässigbar: ein einzelner Testaufruf mit der
  kompletten Ordnerliste (~280 Ordner) lag bei ca. 9K Input-Tokens ≈ unter 1 Cent bei
  Claude Haiku 4.5 ($1/$5 pro 1M Tokens).

## Nextcloud

- Läuft als TrueNAS-Apps-Katalog-App (nicht als eigener Dockge-Stack), Docker-basiert.
- Wurde am 2026-08-28 von Version 31.0.8 (seit 2026-02-28 EOL, keine Sicherheitsupdates
  mehr) sequenziell über 32 → 33 auf **34** aktualisiert (Nextcloud erlaubt keine
  Versions-Sprünge). Vorher wurde ein vollständiges Backup (Daten + Datenbank) erstellt.
- WebDAV-Basis-URL: `https://nextcloud.avernus.cloud/remote.php/dav/files/vault-boy`
- Host-Mount des Webroots: `/mnt/tank/cloud` → `/var/www/html` im Container. **Wichtig:**
  Das Datenverzeichnis wird davon durch einen eigenen, spezifischeren Mount überlagert:
  `/mnt/tank/applications/nextcloud-user-data` → `/var/www/html/data` (bestätigt per
  `occ config:system:get datadirectory` und `docker inspect`). Die eigentlichen
  Nutzerdateien liegen also unter `/mnt/tank/applications/nextcloud-user-data/vault-boy/files/...`
  — **nicht** unter `/mnt/tank/cloud/data/...` wie man vom generischen Webroot-Mount
  naiv annehmen würde. Das ist der Pfad, der DEPOT read-only gemountet wird.
- Der Eingangsordner heißt beim Nutzer exakt **"Scan Eingang"** (mit Leerzeichen, nicht
  "Scan-Eingang" wie im ursprünglichen Plan als Beispielname verwendet) — wichtig für
  `SCAN_EINGANG_LOCAL_PATH`/`SCAN_EINGANG_WEBDAV_PATH` in der `.env`.
- **Seit 2026-09-03 empfohlen (siehe Migrationsschritte unten):** `Scan Eingang` liegt
  INNERHALB von `Dokumente/` (`Dokumente/Scan Eingang`) statt daneben auf oberster Ebene,
  damit alles ein zusammenhängender Baum ist. DEPOT schließt den Scan-Eingang-Pfad
  strukturell und bedingungslos von der Klassifikation aus (unabhängig von
  `excluded_folders`), das ist also gefahrlos.

## DEPOT-Deployment

DEPOT läuft ebenfalls als Dockge-Stack, siehe [`infra/depot/docker-compose.yml`](../infra/depot/docker-compose.yml).

Die `.env` mit den echten Zugangsdaten (Nextcloud-App-Passwort etc.) liegt direkt im
Dockge-Stack-Verzeichnis dieses Stacks (nicht im Git-Repo, da `.env` per `.gitignore`
ausgeschlossen ist und Zugangsdaten niemals ins öffentliche Repo gehören).

### Image-Build via GitHub Actions statt Git-Build-Context (seit 2026-09-03)

**Ursprünglicher Ansatz (verworfen):** GitHub-Repo-URL direkt als Docker-Build-Context
(`build.context: https://github.com/...#master`), damit ein "Rebuild" in Dockge automatisch
den aktuellen `master`-Stand holt, ohne manuelles `git pull` auf dem Server.

**Praxisproblem:** Dockges "Update"-Button führt intern (siehe `backend/stack.ts` im
Dockge-Quellcode) exakt `docker compose pull` gefolgt von `docker compose up -d
--remove-orphans` aus. `docker compose pull` hat aber bei einem Service mit `build:` (statt
`image:`) nichts zu tun — es gibt kein Registry-Image zum Ziehen. Ergebnis: der Button lief
zwar fehlerfrei durch, holte aber nie neuen Code; nötig war stattdessen manuell per SSH
`docker compose build --no-cache && docker compose up -d` im Stack-Ordner (das docker-intern
gecachte Git-Checkout musste zusätzlich mit `--no-cache` umgangen werden, sonst blieb sogar
das ein stiller No-Op).

**Lösung:** [`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml)
baut bei jedem Push auf `master`, der `Dockerfile`, `requirements.txt`, `run.py` oder
`depot/**` verändert, automatisch das Image und pusht es nach GitHub Container Registry
als `ghcr.io/david-stefan-hermann/depot:latest` (zusätzlich mit dem Commit-SHA getaggt).
`infra/depot/docker-compose.yml` referenziert jetzt dieses Image statt eines Build-Contexts.
Damit macht der Dockge-Update-Button genau das, wofür er gebaut ist: `pull` holt das frisch
gebaute Image, `up -d` recreated den Container damit.

**Einmaliger manueller Schritt:** Nach dem ersten erfolgreichen Actions-Lauf muss das
GHCR-Package unter github.com/david-stefan-hermann → Packages → `depot` → Package settings
auf **Public** gestellt werden, sonst kann der TrueNAS-Docker-Daemon es ohne
Registry-Login nicht ziehen (das Repo selbst ist zwar öffentlich, ein frisch erstelltes
GHCR-Package ist es standardmäßig aber nicht).

**Ablauf für ein Update ab jetzt:** Code committen und nach `master` pushen → GitHub Actions
baut automatisch (~1-3 Min, Fortschritt unter dem "Actions"-Tab des Repos einsehbar) → in
Dockge auf den `depot`-Stack den **Update**-Button klicken.

## Migration: Scan Eingang unter Dokumente/, Config-Umbenennung (2026-09-03)

Diese Session hat sowohl Code (Confidence-Cap bei ungültiger Ordnerwahl, Absender-Feld,
`file_into_dokumente`/`save_processed_copy`-Schalter in `DEPOT Config.json`) als auch die
empfohlene Ordnerstruktur geändert. Damit das auf dem echten Server ankommt, sind folgende
manuellen Schritte nötig
(kein Skript, da DEPOT selbst keine Nextcloud-Zugangsdaten in dieser Session hat und ein
automatischer Ordner-Move auf echten Nutzerdaten ohnehin lieber vom Nutzer selbst
gegengeprüft wird):

1. **In Nextcloud:** den Ordner `Scan Eingang` von der obersten Ebene nach `Dokumente/`
   verschieben (Web-UI: verschieben/drag&drop, oder ein WebDAV-`MOVE`). Ergebnis:
   `Dokumente/Scan Eingang`.
2. Innerhalb dieses Ordners den bisherigen `Config`-Unterordner zu `Depot Config`
   umbenennen (falls aus einer früheren Session schon vorhanden — enthält `DEPOT
   Config.json` und die Dateilogs).
3. Falls noch alte Log-Dateien oder eine alte `DEPOT Config.json` direkt in `Scan Eingang`
   (nicht im Unterordner) herumliegen: in `Depot Config` verschieben.
4. Falls bereits Dokumente unter `Dokumente/_Fehlerhaft` quarantänisiert wurden: nach
   `Dokumente/Scan Eingang/Depot Config/_Fehlerhaft` verschieben (neuer Default-Ort, siehe
   unten) — optional, alte Fehlerfälle sind vermutlich sowieso längst manuell bereinigt.
5. **In der `.env` im Dockge-Stack-Verzeichnis** anpassen:
   ```
   SCAN_EINGANG_LOCAL_PATH=/nextcloud-data/Dokumente/Scan Eingang
   SCAN_EINGANG_WEBDAV_PATH=Dokumente/Scan Eingang
   CONFIG_SUBFOLDER=Depot Config
   ```
   `ERROR_FOLDER` NICHT setzen (bzw. die Zeile entfernen, falls vorhanden) — der neue
   Default `<SCAN_EINGANG_WEBDAV_PATH>/<CONFIG_SUBFOLDER>/_Fehlerhaft` greift dann
   automatisch. Kein Docker-Compose-/Volume-Änderung nötig: der Bind-Mount deckt bereits
   den kompletten `files`-Root ab, es ändert sich nur der Unterpfad.
6. In Dockge: **Update** klicken (holt das neue Image mit den Code-Änderungen UND
   übernimmt die geänderte `.env` beim Neustart des Containers).
7. Optional: `file_into_dokumente`/`save_processed_copy` in `DEPOT Config.json` setzen
   (Default entspricht dem bisherigen Verhalten, also nur nötig bei gewünschter
   Abweichung), z.B.:
   ```json
   { "excluded_folders": [], "file_into_dokumente": true, "save_processed_copy": false }
   ```
   Wird bei jeder Datei frisch gelesen — eine Änderung wirkt sofort auf die nächste Datei,
   kein Neustart nötig (anders als die `.env`-Werte oben).
