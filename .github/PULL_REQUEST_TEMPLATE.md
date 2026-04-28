## Zusammenfassung

- Was wurde geaendert?
- Warum ist die Aenderung noetig?

## Checks

- [ ] `python3 -m compileall src tests`
- [ ] `pytest -q`
- [ ] Relevante README-/Dokumentationsstellen aktualisiert
- [ ] Keine Secrets oder privaten Artefakte eingecheckt

## Auswirkungen

- Betroffene Bereiche:
  - [ ] Benchmark Runner
  - [ ] Dashboard
  - [ ] Live Compare
  - [ ] Validation / Scoring
  - [ ] Reports / Exports
  - [ ] Docker / Unraid
  - [ ] CI / Release

## Hinweise fuer Reviewer

- Gibt es neue oder geaenderte Fixture-Dateien?
- Aendert sich ein JSON-/CSV-Schema?
- Gibt es einen manuellen Testpfad, den Reviewer nachstellen sollten?
