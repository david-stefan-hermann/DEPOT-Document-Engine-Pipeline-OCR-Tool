# Infrastruktur-Setup: Ollama auf TrueNAS SCALE

Dieses Dokument hält den tatsächlichen Setup-Verlauf und die Entscheidungen rund um das
lokale LLM (Ollama) auf dem TrueNAS-Server fest, damit sie nicht bei jeder Wartung neu
recherchiert werden müssen.

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
5. **Community-Alternative identifiziert, aber (vorerst) nicht verwendet:**
   [zzzhouuu/truenas-nvidia-drivers](https://github.com/zzzhouuu/truenas-nvidia-drivers)
   bietet inoffizielle Legacy-Treiber-Builds (via `systemd-sysext`) für GTX 700/900/10-Serie.
   Nachteile: kein Build für die exakte Serverversion (25.10.4) verfügbar (nur
   25.10.5/25.10.6/26.0.0-BETA.3 zum Zeitpunkt der Recherche), Treiber-Blob stammt von
   einer privaten Drittanbieter-Domain, muss nach jedem kernel-ändernden TrueNAS-Update
   neu eingerichtet werden.
6. **Entscheidung:** Vorerst **CPU-only** betreiben. Begründung:
   - Der Workload ist Batch-Klassifikation kurzer Texte (keine Echtzeit-Chat-Anfragen),
     bei ca. 8–15 Tokens/Sekunde auf CPU sind das grob 5–15 Sekunden pro Dokument —
     für unregelmäßige Scan-Batches unkritisch.
   - Die GPU brächte keinen garantierten Gewinn (6GB VRAM-Risiko bei Kontext-Overflow
     kann Ollama-Performance laut Community-Benchmarks um das 5–20-fache einbrechen
     lassen) und würde laufenden Wartungsaufwand durch einen inoffiziellen Treiber
     bedeuten.
   - Der Wechsel zu GPU-Beschleunigung bleibt jederzeit möglich, ohne die
     DEPOT-Pipeline selbst anzufassen (siehe unten) — daher bewusst als spätere
     Option offengehalten statt jetzt erzwungen.

## Aktuelles Setup (CPU-only)

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

## Später: Wechsel zu GPU

Sobald GPU-Beschleunigung gewünscht/möglich ist (offizieller TrueNAS-Support für neuere
Baureihen, oder bewusste Entscheidung für den Community-Treiber), reicht es, im
Ollama-Compose die auskommentierte Sektion zu aktivieren:
```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
und den Stack neu zu deployen. Die DEPOT-Pipeline selbst spricht ausschließlich mit der
Ollama-HTTP-API und muss dafür nicht verändert werden.
