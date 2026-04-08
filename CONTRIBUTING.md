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
