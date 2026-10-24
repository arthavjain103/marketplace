# Contributing to the Marketplace Listing Quality Reviewer

Thank you for contributing! This guide covers setting up the repo, making
changes, and submitting pull requests.

## Setup

1. Fork and clone the repo:

   ```bash
   git clone https://github.com/arthavjain103/marketplace.git
   cd marketplace
   ```

2. Install the backend with dev tools:

   ```bash
   uv sync --dev
   ```

   This installs the app plus pytest, ruff, pytest-cov, and httpx.

3. Configure the environment (never commit `.env`):

   ```bash
   cp .env.example .env
   ```

   Fill in `PINECONE_API_KEY`, `PINECONE_INDEX`, and `GROQ_API_KEY` for AI review
   and ingestion. Deterministic checks and the test suite work without them.

4. Install the frontend:

   ```bash
   cd frontend && npm install && cd ..
   ```

## Making changes

1. Create a branch from `main`:

   ```bash
   git checkout -b your-branch-name
   ```

2. Make your changes.

3. Backend: keep code formatted and lint-clean:

   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run ruff format .
   uv run ruff check --fix .
   ```

   Configuration is in `pyproject.toml` under `[tool.ruff]` and
   `[tool.ruff.format]` (line-length 100, double quotes, Python 3.11+).

4. Backend: add or update tests under `tests/` and run them:

   ```bash
   uv run pytest tests/ -q
   uv run pytest tests/ --cov        # coverage must stay >= 60%
   ```

   Integration tests run as part of the default suite. Tests that need real
   model weights opt in via `RUN_MODEL_TESTS=1 uv run pytest tests/ -m
   integration` (downloads models on first run).

5. Frontend: verify the production build:

   ```bash
   cd frontend && npm run build && cd ..
   ```

## Commit and pull request

- Write clear commit messages, preferably conventional style
  (e.g. `fix: ...`, `feat: ...`, `docs: ...`).
- Never commit secrets: `.env`, real API keys, or any credentials must never be
  committed. Only `.env.example` is tracked.
- Open a pull request against `main`. Keep PRs focused on one change.
- CI runs lint, ruff format check, pytest with coverage, pip-audit,
  npm-audit, and gitleaks.

## Releasing

Releases are tagged and versioned in `pyproject.toml`. See `CHANGELOG.md` for
the release history.