# Security Policy

## Supported versions

The project currently treats the latest `main` branch state and the latest tagged
release as supported. If you run an older image, template or config revision, please
upgrade first before reporting a security issue.

## Reporting a vulnerability

Please do **not** open a public GitHub issue for suspected security problems.

Use one of these channels instead:

1. Open a private GitHub security advisory, if enabled for the repository.
2. If that is not available, open a normal issue only for non-sensitive hardening
   questions and omit exploit details, tokens, internal IPs or private prompts.

When you report a security issue, include:

- affected version or commit
- deployment type, for example `docker`, `docker compose` or `unraid`
- whether the issue affects public APIs, dashboard endpoints or local file access
- exact reproduction steps
- sanitized logs or config snippets

## Scope

Please report issues such as:

- authentication or authorization bypass
- exposure of API keys, prompts or benchmark data
- unsafe file access, path traversal or unintended download access
- dashboard endpoints that allow unintended writes or remote command execution
- container defaults that unintentionally weaken isolation

## Hardening expectations

Operators should still follow the normal deployment guidance:

- keep `OPENAI_API_KEY` and other secrets in environment variables, not in committed files
- expose the dashboard only to trusted networks or behind a reverse proxy
- use dedicated persistent directories with controlled permissions
- review exported results before sharing them publicly, because prompts and responses may contain sensitive content
