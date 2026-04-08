# Security Policy

## Verantwortungsbereich

Dieses Projekt verarbeitet Konfigurationsdateien, API-Endpunkte und potenziell sensible Modellantworten. Behandle Benchmarkergebnisse daher wie Anwendungslogs.

## Bitte beachten

- Secrets nur per ENV setzen, nie im Repository.
- `raw_runs.jsonl` und `final_report.json` können sensible Inhalte enthalten.
- Beispielkonfigurationen sind absichtlich mit Platzhaltern versehen.
- Vor Benchmarks mit Kundendaten Prompts und Eingabetexte anonymisieren.

## Meldung von Problemen

Bitte keine Sicherheitsprobleme öffentlich als normales Issue posten, wenn dabei echte Zugangsdaten, interne Endpunkte oder sensible Outputs offengelegt würden. Nutze einen privaten Kontaktweg innerhalb deiner Organisation oder des zuständigen Repo-Owners.
