# LLM Benchmark

Zentraler, Docker- und Unraid-tauglicher Benchmark-Orchestrator fuer OpenAI-kompatible LLM-Endpunkte mit integriertem Web-Dashboard. Das Projekt vergleicht lokale und externe Modelle mit denselben Prompts, schreibt nachvollziehbare Rohdaten, erzeugt Reports fuer Menschen und liefert ein maschinenlesbares `analysis_input.json` fuer spaetere LLM- oder Analysten-Auswertung.

## Zweck

Der produktionsnahe Zielbetrieb in diesem Repository ist:

- lokales Mistral-Modell ueber einen OpenAI-kompatiblen Endpoint
- lokales Qwen-Modell ueber einen OpenAI-kompatiblen Endpoint
- OpenAI als externe Referenz

Das Tool hostet diese Modelle nicht selbst. Es arbeitet als zentraler Benchmark-Orchestrator und kann auf Unraid entweder als:

- persistentes Dashboard laufen
- oder als Job-Container einen Benchmark ausfuehren und danach sauber beenden

## Features

- Benchmarkt mehrere OpenAI-kompatible Endpunkte mit identischen Testfaellen.
- Trennt Warmup-, Cold- und Warm-Runs sauber.
- Bewertet Qualitaet, Format-Treue, Latenz, Stabilitaet, Instruktionsbefolgung und Reproduzierbarkeit.
- Exportiert Rohdaten als JSONL und CSV.
- Exportiert Aggregationen als CSV, Markdown, HTML, `final_report.json` und `analysis_input.json`.
- Liefert ein integriertes Web-Dashboard auf Basis derselben Ergebnisdateien.
- Kann Benchmarks direkt aus dem Dashboard starten und den Fortschritt mit Timeline, Status und Historie anzeigen.
- Enthaelt eine Live-Compare-Ansicht fuer den direkten Antwortvergleich von 2 bis 4 Modellen mit Default auf Mistral, Qwen und OpenAI.
- Fuehrt Live Compare standardmaessig seriell aus, damit CPU-lastige lokale Systeme fairere Laufzeiten liefern.
- Liefert 10 sofort nutzbare Praxis-Presets fuer Docker, Ollama, Paperless, Home Assistant, WhatsApp, YAML, Coding und Loganalyse.
- Zeigt projektbezogene Empfehlungen fuer SecondBrain, secondbrain-voice-gateway und Paperless-KIplus.
- Arbeitet robust weiter, auch wenn einzelne Endpunkte fehlschlagen, timeouts produzieren oder keine Token-Metriken liefern.
- Ist fuer Unraid-Mounts mit `/config`, `/app/tests` und `/app/results` vorbereitet.
- Unterstuetzt einen env-gesteuerten Auto-Run-Startpfad fuer Unraid.

## Dashboard

Das integrierte Dashboard laeuft im selben Container und nutzt denselben Benchmark-Pfad wie die CLI. Es liest die vorhandenen Ergebnisdateien direkt, kann aber zusaetzlich kontrolliert einen neuen Benchmark im Hintergrund starten und dessen Status sichtbar machen.

Zusaetzlich gibt es unter `/live-compare` eine interaktive Live-Compare-Ansicht. Sie verwendet dieselben Modellkonfigurationen und denselben OpenAI-kompatiblen Client-Stack wie der Benchmark, sendet aber eine freie Eingabe direkt an die ausgewaehlten Modelle und zeigt die Antworten nebeneinander an.

Wichtige Fairness-Regel fuer produktive CPU-Systeme:

- Standardmodus ist `Fair Compare (seriell)`.
- Die Modelle laufen dabei in dieser Reihenfolge: `Mistral Local`, `Qwen Local`, `OpenAI Reference`.
- Das vermeidet, dass zwei lokale Modelle gleichzeitig um dieselbe CPU-Zeit konkurrieren und dadurch scheinbar langsamer oder schneller wirken, als sie isoliert waeren.
- `Parallel Compare` bleibt verfuegbar, ist aber sichtbar als potenziell unfair fuer CPU-only- oder CPU-lastige Server gekennzeichnet.

Vor dem eigentlichen Benchmark kann das Dashboard ausserdem per Button einen leichten LLM-Erreichbarkeits-Check ausfuehren. Dieser prueft fuer alle aktiven Modelle den OpenAI-kompatiblen `/models`-Endpunkt, inklusive Auth und Sichtbarkeit des konfigurierten Modellnamens.

Wichtiger Praxis-Hinweis fuer Qwen auf Ollama:

- Wenn ein Qwen-Thinking-Modell ueber Ollamas OpenAI-kompatible `/v1/chat/completions`-Route angebunden wird, sollte in der Modellkonfiguration fuer produktionsnahe Benchmarks meist `default_parameters.reasoning_effort: "none"` gesetzt werden.
- Sonst kann der Endpoint bei kleineren `max_tokens`-Budgets nur `reasoning` liefern und ein leeres Assistant-`content` zurueckgeben.
- Das Benchmark-Tool erkennt diesen Fall inzwischen explizit als `empty_assistant_content`, statt ihn still als normalen Validierungsfehler zu behandeln.

Verfuegbare Routen:

- `/dashboard`
- `/live-compare`
- `/health`
- `/api/dashboard/run/current`
- `/api/dashboard/run/history`
- `/api/dashboard/run/start`
- `/api/dashboard/connectivity/current`
- `/api/dashboard/connectivity/check`
- `/api/dashboard/summary`
- `/api/dashboard/models`
- `/api/dashboard/categories`
- `/api/dashboard/tests`
- `/api/dashboard/failures`
- `/api/dashboard/domain`
- `/api/dashboard/live-compare`
- `/api/dashboard/live-compare/current`
- `/api/dashboard/live-compare/history`
- `/api/dashboard/live-compare/<run_id>`
- `/downloads/<datei>`

Das Dashboard orientiert sich optisch am Synthwave-/Dark-Design des Feberdin-Templates:

- dunkle Panels mit klarer Lesbarkeit
- starke Kartenhierarchie
- farbige Status-Badges
- kontrastreiche Tabellen
- projektbezogene Empfehlungskarten

## Repo-spezifische Suiten

Zusatz zu `core`, `api_replacement`, `long_context` und `quick_compare`:

- `secondbrain`
- `voice_gateway`
- `paperless_kiplus`

### SecondBrain

Die Suite [secondbrain_suite.yaml](/Users/joachim.stiegler/LLM-Benchmark/fixtures/suites/secondbrain_suite.yaml) testet unter anderem:

- RAG / Kontexttreue
- quellenorientierte Antworten
- strukturierte Query-JSONs
- Dokument- und Mail-Zusammenfassungen
- Fakt- und Metadatenextraktion
- Prompt-Injection-Abwehr
- Home-Assistant-Kontext
- WhatsApp-Kommunikation
- Banking- und Salary-Kontext

### secondbrain-voice-gateway

Die Suite [voice_gateway_suite.yaml](/Users/joachim.stiegler/LLM-Benchmark/fixtures/suites/voice_gateway_suite.yaml) testet unter anderem:

- Routing-Klassifikation
- Prefix-Priorisierung
- kurze Alexa-Antworten
- Backend-/Tool-Selektion
- Ambiguitaetsbehandlung
- Home-Assistant-Action-Safety
- Docker-Ops-Sprachantworten
- Troubleshooting-Kurzantworten
- Latest-Mail-Shortcut

### Paperless-KIplus

Die Suite [paperless_kiplus_suite.yaml](/Users/joachim.stiegler/LLM-Benchmark/fixtures/suites/paperless_kiplus_suite.yaml) testet unter anderem:

- Dokumentklassifikation
- strikte JSON-/Schema-Treue
- YAML-Generierung
- Precheck-/Skip-/Duplicate-Entscheidungen
- AI-Notes
- produktive Runtime-Defaults
- Tax Enrichment
- Review-Flags und Unsicherheit
- sparsame Tagging-Empfehlungen

## Architektur

Das Projekt bleibt bewusst in einer einzigen Architektur:

1. `config`
   Laedt YAML/JSON-Konfigurationen und expandiert `${ENV_VAR}`-Platzhalter.
2. `domain`
   Enthaelt die stabilen Typmodelle fuer Testfaelle, Responses und Ergebnisdatensaetze.
3. `clients`
   Spricht OpenAI-kompatible Chat-Completions-Endpunkte an, inklusive Retry-Logik und optionaler TTFT-Messung.
4. `validation`
   Prueft JSON, YAML, JSON-Schema, Pflichtfelder, Tool-Calls und Instruktionsregeln transparent.
5. `runner`
   Orchestriert Wiederholungen, Warmups, Fehlerisolation und Scoring.
6. `reporting`
   Baut CSV-, Markdown-, HTML-, Final-JSON- und Analyse-JSON-Reports.
7. `dashboard`
   Serviert das Web-Dashboard direkt aus den vorhandenen Ergebnisdateien, kann neue Runs im Hintergrund starten und bietet zusaetzlich Live Compare fuer direkte Modellantworten.

Wichtige Architekturentscheidung:

- Kein zweiter Datenpfad fuer das Dashboard
  Das Dashboard liest `final_report.json`, `analysis_input.json` und `raw_runs.jsonl` direkt. Dadurch bleiben CLI, Reports und Web immer konsistent.

## Projektstruktur

```text
.
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CHANGELOG.md
├── docker
│   └── entrypoint.sh
├── fixtures
│   ├── config
│   │   ├── config.example.yaml
│   │   └── config.unraid.example.yaml
│   ├── suites
│   │   ├── quick_compare.yaml
│   │   ├── secondbrain_suite.yaml
│   │   ├── voice_gateway_suite.yaml
│   │   └── paperless_kiplus_suite.yaml
│   ├── live_compare
│   │   └── presets.yaml
│   └── tests
│       ├── 001_chat_cap_theorem.yaml
│       ├── ...
│       ├── 101_quick_chat_triage.yaml
│       ├── ...
│       └── domain
│           ├── secondbrain
│           ├── voice_gateway
│           └── paperless_kiplus
├── src
│   └── llm_benchmark
│       ├── cli.py
│       ├── clients
│       ├── config
│       ├── dashboard
│       │   ├── app.py
│       │   ├── service.py
│       │   ├── templates
│       │   └── static
│       ├── domain
│       ├── reporting
│       ├── runner
│       ├── templates
│       ├── validation
│       └── utils.py
├── tests
│   ├── test_loader_and_validation.py
│   ├── test_scoring_and_reporting.py
│   └── test_dashboard_service.py
└── unraid
    └── llm-benchmark.xml
```

## Konservative Defaults fuer aeltere Unraid-Server

Die produktionsnahe Unraid-Beispielkonfiguration unter [config.unraid.example.yaml](/Users/joachim.stiegler/LLM-Benchmark/fixtures/config/config.unraid.example.yaml) nutzt bewusst konservative Defaults:

- `concurrency: 1`
  Grund: Grosse lokale Modelle blockieren auf CPU-Systemen sonst schnell das ganze System.
- `warmup_runs: 1`
  Grund: Der erste echte Vergleich soll Cache- und Initialisierungseffekte sichtbar machen, aber nicht unnoetig Zeit verbrennen.
- `default_repetitions: 2`
  Grund: Reproduzierbarkeit wird sichtbar, ohne dass die Laufzeit explodiert.
- `default_timeout_seconds: 300`
  Grund: Lokale CPU-Inferenz kann deutlich langsamer sein als Cloud-APIs.
- `max_retries: 2`
  Grund: Genug fuer kurzzeitige Netzwerk- oder Busy-Fehler, aber nicht so viel, dass instabile Endpunkte schoen gerechnet werden.
- `retry_backoff_seconds: 2.0`
  Grund: Exponentieller Backoff von 2s auf 4s und 8s ist defensiv fuer lokale und externe Endpunkte.
- `temperature: 0.0` und `top_p: 1.0`
  Grund: Benchmarking soll moeglichst deterministisch und fair bleiben.
- `stream_for_ttft: false`
  Grund: Nicht alle lokalen OpenAI-kompatiblen Runtimes liefern Streaming-Telemetrie konsistent.
- `max_output_tokens`
  Kurztests typischerweise 90 bis 180, strukturierte Tests 120 bis 260, YAML-Test bewusst hoeher.

## Unraid Ordnerstruktur

Empfohlene Host-Struktur:

```text
/mnt/user/appdata/llm-benchmark/
├── config/
│   └── config.unraid.example.yaml
├── results/
│   ├── raw_runs.jsonl
│   ├── raw_runs.csv
│   ├── summary_by_model.csv
│   ├── summary_by_category.csv
│   ├── final_report.json
│   ├── final_report.md
│   ├── final_report.html
│   └── analysis_input.json
├── tests/
│   ├── 001_chat_cap_theorem.yaml
│   ├── ...
│   └── domain/
│       ├── secondbrain/
│       ├── voice_gateway/
│       └── paperless_kiplus/
├── suites/
│   ├── quick_compare.yaml
│   ├── secondbrain_suite.yaml
│   ├── voice_gateway_suite.yaml
│   └── paperless_kiplus_suite.yaml
└── logs/
```

### Welche Dateien wohin gehoeren

- `config/`
  Persistente YAML-Konfigurationen. Diese Dateien passt der Benutzer aktiv an.
- `results/`
  Vom Container erzeugte Reports und Rohdaten. Dieser Ordner muss persistent sein.
- `tests/`
  Ausfuehrbare Benchmark-Testdateien plus synthetische Domain-Fixtures. Dieser Ordner sollte persistent sein.
- `suites/`
  Suite-Manifeste zur Dokumentation und internen Versionierung.
- `logs/`
  Optional. Standardmaessig loggt der Container nach STDOUT.

### Welche Dateien persistent sein muessen

- `/mnt/user/appdata/llm-benchmark/config`
- `/mnt/user/appdata/llm-benchmark/results`
- `/mnt/user/appdata/llm-benchmark/tests`

### Welche Dateien der Container erzeugt

- `raw_runs.jsonl`
- `raw_runs.csv`
- `summary_by_model.csv`
- `summary_by_category.csv`
- `final_report.json`
- `final_report.md`
- `final_report.html`
- `analysis_input.json`

### Welche Dateien der Benutzer anpassen soll

- `config/config.unraid.example.yaml` oder eine abgeleitete eigene Config
- eigene oder angepasste Testfaelle unter `tests/`
- optional Suite-Manifeste unter `suites/`

## Unraid Template XML

Die Vorlage liegt unter [llm-benchmark.xml](/Users/joachim.stiegler/LLM-Benchmark/unraid/llm-benchmark.xml).

Wichtige Eigenschaften:

- direkte Mounts fuer `/config`, `/app/results`, `/app/tests`
- Repository in der XML steht bewusst auf `llm-benchmark:local`, weil das Unraid-Template fuer ein lokal gebautes Image gedacht ist
- optionaler Dashboard-Port `8080`
- `BENCHMARK_ACTION=dashboard` als sinnvoller Default fuer persistenten Betrieb
- `BENCHMARK_ACTION=run` fuer One-Shot-Benchmark-Jobs
- WebUI zeigt auf `/dashboard`

## Beispielkonfiguration fuer 3 Modelle

Die produktionsnahe Drei-Modell-Konfiguration liegt unter [config.unraid.example.yaml](/Users/joachim.stiegler/LLM-Benchmark/fixtures/config/config.unraid.example.yaml).

Sie enthaelt:

- `mistral_local`
- `qwen_local`
- `openai_reference`

Dort ist klar kommentiert:

- wo `model_name` geaendert wird
- wo `base_url` geaendert wird
- wie ein Modell per `enabled: false` deaktiviert wird
- wie `timeout_seconds` pro Modell gesetzt werden
- wie globale und modellbezogene `default_parameters` ueberschrieben werden

## Ergebnisdateien

Nach einem Lauf entstehen mindestens:

- `results/raw_runs.jsonl`
- `results/raw_runs.csv`
- `results/summary_by_model.csv`
- `results/summary_by_category.csv`
- `results/final_report.json`
- `results/final_report.md`
- `results/final_report.html`
- `results/analysis_input.json`

Zusaetzlich erzeugt die Live-Compare-Ansicht eigene persistente Dateien unter:

- `results/live_compare/current_state.json`
- `results/live_compare/history.json`
- `results/live_compare/runs/<run_id>.json`

Diese Dateien enthalten jetzt auch:

- `execution_mode`
- `execution_order`
- `manual_note`
- `queue_wait_ms`
- `execution_start_at`
- `execution_end_at`
- `isolated_duration_ms`
- `total_elapsed_since_run_start_ms`

Fuer normale Benchmarklaeufe gibt es zusaetzlich pro Run eigene Snapshot-Artefakte unter:

- `results/history/<benchmark_run_id>/raw_runs.jsonl`
- `results/history/<benchmark_run_id>/raw_runs.csv`
- `results/history/<benchmark_run_id>/summary_by_model.csv`
- `results/history/<benchmark_run_id>/summary_by_category.csv`
- `results/history/<benchmark_run_id>/final_report.json`
- `results/history/<benchmark_run_id>/final_report.md`
- `results/history/<benchmark_run_id>/final_report.html`
- `results/history/<benchmark_run_id>/analysis_input.json`

Das Dashboard zeigt diese Snapshots jetzt direkt im Run-Verlauf als separate Download-Links an. Damit kannst du einzelne Laeufe gezielt sichern oder teilen, ohne die aktuellen Hauptdateien manuell kopieren zu muessen.

### `final_report.json`

Dieses Artefakt ist fuer Menschen und strukturierte Tools gedacht und enthaelt unter anderem:

- `benchmark_info`
- `environment_info`
- `models`
- `test_suites`
- `aggregate_scores`
- `rankings`
- `strengths`
- `weaknesses`
- `anomalies`
- `failed_runs`
- `recommendations`

### `analysis_input.json`

Dieses Artefakt ist speziell als Input fuer ein spaeteres Analyse-LLM gedacht. Es enthaelt jetzt unter anderem:

- `benchmark_metadata`
- `hardware_and_environment_hints`
- `model_rankings`
- `speed_summary`
- `category_findings`
- `aggregate_scores`
- `repo_recommendations`
- `best_model_for_secondbrain`
- `best_model_for_voice_gateway`
- `best_model_for_paperless_kiplus`
- `security_behavior_summary`
- `structured_output_summary`
- `voice_response_summary`
- `tax_enrichment_summary`
- `representative_failures`
- `representative_success_examples`
- `model_cards`

## Beispielbefehle

Config validieren:

```bash
benchmark validate-config --config /config/config.unraid.example.yaml
```

Modelle auflisten:

```bash
benchmark list-models --config /config/config.unraid.example.yaml
```

Tests der Quick-Compare-Suite auflisten:

```bash
benchmark list-tests --tests-dir /app/tests --suite quick_compare
```

SecondBrain-Suite auflisten:

```bash
benchmark list-tests --tests-dir /app/tests --suite secondbrain
```

Quick Compare ausfuehren:

```bash
benchmark run --config /config/config.unraid.example.yaml --suite quick_compare --tests-dir /app/tests --results-dir /app/results
```

SecondBrain-Suite ausfuehren:

```bash
benchmark run --config /config/config.unraid.example.yaml --suite secondbrain --tests-dir /app/tests --results-dir /app/results
```

Voice-Gateway-Suite ausfuehren:

```bash
benchmark run --config /config/config.unraid.example.yaml --suite voice_gateway --tests-dir /app/tests --results-dir /app/results
```

Paperless-KIplus-Suite ausfuehren:

```bash
benchmark run --config /config/config.unraid.example.yaml --suite paperless_kiplus --tests-dir /app/tests --results-dir /app/results
```

Report aus vorhandenem JSONL neu erzeugen:

```bash
benchmark report --input /app/results/raw_runs.jsonl --output /app/results
```

Dashboard starten:

```bash
benchmark dashboard --config /config/config.unraid.example.yaml --tests-dir /app/tests --results-dir /app/results --host 0.0.0.0 --port 8080
```

Live Compare im Browser nutzen:

```text
http://<host>:8080/live-compare
```

## Docker Quickstart

### 1. Installation

```bash
python3 -m venv .venv
. .venv/bin/activate
.venv/bin/pip install -e ".[dev]"
```

### 2. Tests

```bash
.venv/bin/python -m pytest -q
```

### 3. Dashboard lokal

```bash
benchmark dashboard --config fixtures/config/config.example.yaml --tests-dir fixtures/tests --results-dir results --port 8080
```

### 4. Docker Compose

Das Compose-Setup startet standardmaessig das Dashboard:

```bash
docker compose up -d llm-benchmark
```

Benchmark-Lauf parallel oder manuell:

```bash
docker compose run --rm llm-benchmark run --config /config/config.example.yaml --suite quick_compare --tests-dir /app/tests --results-dir /app/results
```

## Vorbereitung auf Unraid

### 1. Verzeichnisse anlegen

```bash
mkdir -p /mnt/user/appdata/llm-benchmark/config
mkdir -p /mnt/user/appdata/llm-benchmark/results
mkdir -p /mnt/user/appdata/llm-benchmark/tests
mkdir -p /mnt/user/appdata/llm-benchmark/suites
mkdir -p /mnt/user/appdata/llm-benchmark/logs
```

### 2. Config-Datei einfuegen

Kopiere [config.unraid.example.yaml](/Users/joachim.stiegler/LLM-Benchmark/fixtures/config/config.unraid.example.yaml) nach:

```text
/mnt/user/appdata/llm-benchmark/config/config.unraid.example.yaml
```

Passe dann mindestens an:

- `MISTRAL_BASE_URL`
- `QWEN_BASE_URL`
- `MISTRAL_MODEL_NAME`
- `QWEN_MODEL_NAME`
- optional `OPENAI_MODEL_NAME`

### 3. Tests und Suiten einfuegen

Kopiere:

- Inhalte aus `fixtures/tests/` nach `/mnt/user/appdata/llm-benchmark/tests/`
- Inhalte aus `fixtures/suites/` nach `/mnt/user/appdata/llm-benchmark/suites/`

Fuer den ersten produktiven Lauf reichen die fuenf `quick_compare`-Dateien. Fuer projektspezifische Entscheidungen kopierst du zusaetzlich die drei Domain-Unterordner unter `tests/domain/`.

### 4. OpenAI API Key setzen

Im Unraid-Template oder Container-Setup:

- `OPENAI_API_KEY` setzen
- `OPENAI_BASE_URL` nur aendern, wenn du nicht direkt gegen OpenAI testest

### 5. Container als Dashboard starten

Importiere [llm-benchmark.xml](/Users/joachim.stiegler/LLM-Benchmark/unraid/llm-benchmark.xml) oder uebernimm dessen Felder manuell.

Wichtig:

- Die XML erwartet aktuell ein lokal gebautes Image `llm-benchmark:local`.
- Falls du das Repo direkt auf Unraid gebaut hast, passt das bereits.
- Falls du spaeter ein eigenes Registry-Image nutzt, kannst du das Repository-Feld im Template entsprechend ueberschreiben.

Fuer den persistenten Dashboard-Betrieb:

- `BENCHMARK_AUTO_RUN=true`
- `BENCHMARK_ACTION=dashboard`
- `BENCHMARK_CONFIG=/config/config.unraid.example.yaml`
- `DASHBOARD_PORT=8080`

Dann ist das Dashboard unter `http://<unraid-ip>:8080/dashboard` erreichbar.

### 6. Ersten Benchmark-Lauf ausfuehren

Bevorzugter Weg auf Unraid:

1. Dashboard unter `http://<unraid-ip>:8080/dashboard` oeffnen.
2. Im Bereich `Run Control` zuerst optional `LLM-Erreichbarkeit pruefen` klicken.
3. Danach die Suite `quick_compare` waehlen.
4. `Benchmark starten` klicken.
5. Timeline, Fortschrittsbalken und Historie im selben Webinterface verfolgen.
6. Nach Abschluss aktualisiert das Dashboard die Reports automatisch.

Alternative fuer CLI/Terminal:

```bash
benchmark run --config /config/config.unraid.example.yaml --suite quick_compare --tests-dir /app/tests --results-dir /app/results
```

Hinweis:

- `docker run --rm ... run ...` ist weiterhin moeglich, aber unkomfortabler fuer Unraid, weil der Job-Container nach Abschluss sofort verschwindet.
- Fuer produktiven Betrieb ist `BENCHMARK_ACTION=dashboard` als dauerhafter Container die angenehmere Variante.

### 6a. Live Compare direkt im Dashboard nutzen

1. `http://<unraid-ip>:8080/live-compare` oeffnen.
2. Oben eine echte Alltagsfrage oder ein Log/Prompt eintragen.
3. Optional einen kurzen System-Prompt hinterlegen.
4. Standardmaessig bleiben `Mistral Local`, `Qwen Local` und `OpenAI Reference` aktiviert.
5. Einen Ausfuehrungsmodus waehlen. Standard ist `Fair Compare (seriell)`.
6. Optional eine kurze manuelle Notiz wie `Mistral wirkt vorsichtiger bei Faktenfragen` hinterlegen.
7. `Alle 3 vergleichen` klicken.
8. Die drei Antwortkarten zeigen live Status, isolierte Dauer, Queue-Wartezeit, Token-Metriken, Fehler und die eigentliche Antwort.
9. Fruehere Vergleiche bleiben unter `results/live_compare/` und in der History auf der Seite erhalten.

### 7. Ergebnisse finden

Die Artefakte liegen unter:

```text
/mnt/user/appdata/llm-benchmark/results/
```

Besonders wichtig:

- `summary_by_model.csv`
- `summary_by_category.csv`
- `final_report.html`
- `final_report.json`
- `analysis_input.json`
- `history/<benchmark_run_id>/...` fuer laufspezifische Snapshots und Downloads
- `live_compare/history.json`
- `live_compare/runs/<run_id>.json`

### 8. Reports interpretieren

Praktische Reihenfolge:

1. Dashboard `Overview` fuer schnellen Gesamtblick
2. `Models` fuer Score-Karten, Median und p95
3. `Domains` fuer projektbezogene Empfehlungen
4. `Failures` fuer Timeouts, Schema-Drift und Sicherheitsprobleme
5. `analysis_input.json`, wenn spaeter ein anderes LLM die Ergebnisse auswerten soll
6. `Live Compare`, wenn du vor einer echten Modellentscheidung die Antwortqualitaet direkt nebeneinander sehen willst

## Live Compare

Die Live-Compare-Ansicht ist fuer die schnelle Alltagsentscheidung gedacht: "Welches Modell wuerde ich fuer genau diese konkrete Frage bevorzugen?"

Standardmaessig werden diese drei Modelle vorausgewaehlt:

- `Mistral Local`
- `Qwen Local`
- `OpenAI Reference`

### Fair Compare vs. Parallel Compare

Der Standardmodus ist jetzt bewusst:

- `Fair Compare (seriell)`

Warum:

- Auf CPU-lastigen lokalen Servern verfaelschen parallele Requests die Laufzeiten stark.
- Zwei lokale Modelle, die gleichzeitig rechnen, teilen sich dieselben CPU-Ressourcen und wirken dadurch kuenstlich langsamer.
- Serielle Ausfuehrung liefert fuer `isolated_duration_ms` die deutlich belastbarere Vergleichsmetrik.

Verfuegbare Modi:

- `Fair Compare (seriell)`
  Modelle laufen nacheinander in der festen Reihenfolge `Mistral Local`, `Qwen Local`, `OpenAI Reference`.
- `Parallel Compare`
  Alle ausgewaehlten Modelle starten moeglichst gleichzeitig. Das ist praktisch fuer grobe Antwortvergleiche, aber fuer lokale CPU-Latenzen oft unfair.

Die Ansicht zeigt pro Modellspalte:

- Status `waiting`, `running`, `finished` oder `failed`
- Start- und Endzeit
- `isolated_duration_ms` als wichtigste faire Laufzeitmetrik
- `queue_wait_ms`
- `total_elapsed_since_run_start_ms`
- optional `ttft_ms`
- optional `input_tokens`, `output_tokens` und `tokens_per_second`
- HTTP-Status
- optionale JSON-Vorschau bei strukturierter Antwort
- kompakte Fehlerkarte mit einklappbaren technischen Details
- Quick-Badges wie `fastest`, `shortest-response`, `json-detected`, `yaml-detected`, `code-detected` oder `uncertainty-marked`

### Die 10 integrierten Praxis-Presets

Die Presets liegen unter [presets.yaml](/Users/joachim.stiegler/LLM-Benchmark/fixtures/live_compare/presets.yaml) und koennen direkt in das Prompt-Feld geladen werden.

1. `Faktenfrage mit Unsicherheitsdisziplin`
   Prueft, ob ein Modell sauber zwischen Wissen und Unsicherheit trennt.
2. `Docker-Fehleranalyse`
   Praxisnaher GHCR-/Docker-/Unraid-Support-Fall.
3. `Ollama Troubleshooting`
   Strukturierte Fehlersuche fuer lokale CPU-only-Inferenz.
4. `Paperless Dokumentklassifikation`
   JSON-Treue fuer dokumentnahe Workflows.
5. `Steuerliche Einordnung vorsichtig`
   Konservative Tax-Enrichment-Naeherung ohne ueberzogene Sicherheit.
6. `Home Assistant Statusinterpretation`
   Kurze, sprachgeeignete Zusammenfassung fuer Voice-/HA-Szenarien.
7. `WhatsApp Gespraechsauswertung`
   Kommunikations- und SecondBrain-nahe Extraktion.
8. `YAML Konfiguration robust`
   Strukturausgabe und Konfigurationsdisziplin ohne Markdown-Huelle.
9. `Python Coding Test`
   Kleine Coding-Aufgabe mit Funktion und pytest-Test.
10. `Loganalyse mit Ursache und Massnahmen`
    Strukturierte Root-Cause- und Next-Step-Antwort fuer echte Betriebsfehler.

Persistenz:

- Jeder Live-Compare-Lauf wird als eigenes JSON unter `results/live_compare/runs/` gespeichert.
- Die History-Liste auf der Seite liest diese lauffaehigen Persistenzdaten wieder ein.
- Eine optionale manuelle Notiz wird zusammen mit dem Lauf gespeichert.

Grenzen der Aussagekraft:

- Live Compare ist absichtlich weniger streng als der Benchmark und bewertet nicht automatisch dieselbe Regeltiefe wie eine Test-Suite.
- Lokale CPU-Modelle koennen im parallelen Direktvergleich deutlich langsamer sein als bei einem isolierten Einzelrequest.
- Fuer CPU-lastige lokale Hardware ist `Fair Compare (seriell)` die empfohlene Standardwahl.
- Token- und TTFT-Metriken sind providerabhaengig und nicht bei jedem Endpoint verfuegbar.
- Ein besonders gutes Live-Compare-Ergebnis ersetzt keine projektspezifische Suite wie `secondbrain`, `voice_gateway` oder `paperless_kiplus`.

## Wie die projektbezogenen Rankings zu lesen sind

- `SecondBrain`
  Achte besonders auf Kontexttreue, Zitationsdisziplin, Sicherheitsgehorsam und konservative Extraktion.
- `secondbrain-voice-gateway`
  Achte besonders auf exakte Routing-Disziplin, kurze sprechbare Antworten und sichere Allowlist-Beachtung.
- `Paperless-KIplus`
  Achte besonders auf strikte JSON-/YAML-Treue, konservative Entscheidungen, Review-Flags und Tax-Enrichment-Sicherheit.

Ein hohes Gesamtranking allein reicht nicht. Fuer produktive Auswahlentscheidungen solltest du immer:

- die jeweilige Repo-Suite
- die Failure-Ansicht
- die strukturierten Teilzusammenfassungen in `analysis_input.json`

gemeinsam lesen.

## Grenzen der Vergleichbarkeit

- Identische Prompts und Token-Budgets machen den Vergleich fairer, aber nicht perfekt.
- Externes Prompt-Caching oder providerseitige Optimierungen koennen Antworten und Latenz beeinflussen.
- OpenAI-Modelle liefern mitunter stabilere Token-Metriken als lokale OpenAI-kompatible Runtimes.
- Lokale CPU-basierte Inferenz kann stark schwankende Latenzen erzeugen.
- Fehlende Token-Metriken werden nicht geraten, sondern bleiben leer.
- Streaming/TTFT ist optional und nur dann aussagekraeftig, wenn alle verglichenen Backends aehnlich streamen.

## Troubleshooting

- `Config validation failed`
  Pruefe Pfade, `${ENV_VAR}`-Platzhalter und Modell-IDs.
- `Tests directory does not exist`
  Pruefe den Mount fuer `/app/tests`.
- Run startet im Dashboard nicht
  Pruefe die Preflight-Hinweise im Bereich `Run Control`. Ein leeres Unraid-Mount auf `/app/tests` ueberdeckt die eingebauten Fixture-Tests im Container.
- `Missing required API key environment variables`
  Pruefe `OPENAI_API_KEY` oder andere referenzierte Secret-Variablen.
- Viele `json_parse_failed` oder `yaml_parse_failed`
  Das Modell verletzt die geforderte Format-Treue; schaue in `raw_runs.jsonl`, `final_report.json` und Dashboard `Tests`.
- Lokales Modell ist sehr langsam
  Erhoehe nicht zuerst die Parallelisierung. Pruefe lieber Quantisierung, Runtime-Flags, CPU-Last und `timeout_seconds`.
- Dashboard zeigt keine Daten
  Pruefe, ob `final_report.json`, `analysis_input.json` und `raw_runs.jsonl` in `/app/results` liegen oder starte den ersten Lauf direkt im Dashboard.

## Logs und Debugging

- CLI und Container verstehen `--debug` bzw. `BENCHMARK_DEBUG=true`
- Standard-Log-Level ueber `LOG_LEVEL=INFO` oder `LOG_LEVEL=DEBUG`
- Wichtige Laufzeitdetails landen in:
  - Container-Logs
  - `raw_runs.jsonl`
  - `final_report.json`
  - `analysis_input.json`

## Security-Hinweise

- API-Keys nie in YAML committen. Verwende `api_key_env` plus Umgebungsvariable.
- Lokale Modellendpunkte sollten nur im vertrauenswuerdigen Netz erreichbar sein.
- Das Dashboard ist fuer internes Betriebsnetz gedacht und sollte nicht ohne Reverse Proxy, Authentifizierung und Netzwerkgrenzen oeffentlich exponiert werden.
- Secrets werden in Umgebungs-Metadaten nur maskiert dargestellt.

## Lizenz

Siehe [LICENSE](/Users/joachim.stiegler/LLM-Benchmark/LICENSE).
