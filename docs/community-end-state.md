# Community End State

## Zweck dieses Dokuments

Dieses Dokument beschreibt den angestrebten stabilen Ausgangszustand fuer die erste oeffentliche Community-Version. Es soll Mitwirkenden schnell erklaeren, was als "fertig genug fuer reale Nutzung" gilt und welche Erweiterungen darauf sauber aufbauen koennen.

## Zielbild

Das Projekt soll gleichzeitig:

- auf Unraid produktiv nutzbar sein
- lokal per Docker oder Python reproduzierbar laufen
- fuer neue Mitwirkende schnell verstehbar sein
- strukturierte Benchmark-Artefakte fuer spaetere Analyse liefern
- ohne Parallelarchitektur sowohl CLI als auch Dashboard bedienen

## Bestandteile des Community-faehigen Endstands

### 1. Stabile Kernfunktion

- mehrere OpenAI-kompatible Endpunkte benchmarken
- Ergebnisse reproduzierbar persistieren
- klare Validation- und Scoring-Regeln
- robuste Fehlerdarstellung statt stiller Ausfaelle

### 2. Community-taugliche Distribution

- oeffentliches GHCR-Image
- Dockerfile mit OCI-Metadaten
- Unraid-Template, das auf das veroeffentlichte Image zeigt
- Compose-Setup fuer lokale oder labnahe Nutzung

### 3. Kollaborationsfaehigkeit

- CI fuer Tests und Container-Build
- Issue-Templates und PR-Template
- Contributing- und Security-Hinweise
- klarer Code of Conduct

### 4. Erweiterbarkeit

- neue Suiten koennen als Fixture-Dateien ergaenzt werden
- neue Modellprovider sollen ueber den vorhandenen OpenAI-kompatiblen Pfad oder kleine Adapter folgen
- Dashboard und Reports bleiben an denselben Ergebnisdateien gekoppelt

## Was bewusst nicht Teil des Endstands ist

- eigenes Modellhosting
- Multi-Node-Scheduling
- verteilte Worker
- automatisches Benchmarking mit echten privaten Produktionsdaten
- serverseitige Benutzerverwaltung fuer das Dashboard

## Empfohlene Weiterentwicklungsrichtungen

1. Weitere repo-spezifische Suiten
2. Export-Integrationen in BI- oder Notebook-Workflows
3. Optionaler SQLite- oder DuckDB-Layer fuer groeßere historische Analysen
4. Diff- und Consensus-Features fuer Live Compare
5. zusätzliche Provider-spezifische Guardrails fuer Structured Output
