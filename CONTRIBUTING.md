# Contributing

Danke für Beiträge. Dieses Repository bevorzugt kleine, lesbare Änderungen mit klarer Fehlerbehandlung und guter Dokumentation.

## Entwicklungsumgebung

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```

## Lokale Checks

```bash
pytest -q
benchmark validate-config --config fixtures/config/config.example.yaml
benchmark list-tests --tests-dir fixtures/tests
```

## Arbeitsweise

- Änderungen klein und nachvollziehbar halten.
- Öffentliche JSON- und CSV-Felder nur bewusst ändern, weil externe Analyse darauf aufbauen kann.
- Neue Testfälle immer mit klarer `validation_rules`-Definition anlegen.
- Bei neuen Provider-Sonderfällen zuerst das Verhalten dokumentieren, dann abstrahieren.
- Vor größeren Änderungen kurz in [community-end-state.md](/Users/joachim.stiegler/LLM-Benchmark/docs/community-end-state.md) prüfen, ob der Vorschlag den stabilen Kern oder eher eine spätere Erweiterung betrifft.

## Stil

- Python 3.12
- Striktes Typing
- Klare Namen statt cleverer Einzeiler
- Kommentare erklären Absicht und Debug-Pfade
- Keine still geschluckten Exceptions

## Pull Requests

- Kurz beschreiben, was geändert wurde.
- Relevante Testfälle oder neue Fixtures nennen.
- Falls Reporting-Struktur geändert wurde, die Auswirkungen auf `final_report.json` explizit erwähnen.
- Die PR-Checkliste aus [.github/PULL_REQUEST_TEMPLATE.md](/Users/joachim.stiegler/LLM-Benchmark/.github/PULL_REQUEST_TEMPLATE.md) vollständig durchgehen.
- Bei Community-Diskussionen gilt der Ton aus [CODE_OF_CONDUCT.md](/Users/joachim.stiegler/LLM-Benchmark/CODE_OF_CONDUCT.md).

## CI und Releases

- Jeder PR sollte lokal mindestens `python3 -m compileall src tests` und `pytest -q` bestehen.
- GitHub Actions bauen zusätzlich das Container-Image und prüfen die Beispiel-Config.
- Das öffentliche Container-Image wird über GHCR veröffentlicht und ist kein separater Hand-Build.
