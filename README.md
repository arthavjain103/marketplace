# Marketplace Listing Quality Reviewer

Reviews product/service listings against a marketplace policy and brand-content guide.

**Deterministic rule checks (no AI) + RAG + LLM review**, with a reviewer workflow
(approve / edit / reject), a compare view, batch processing, an append-only audit
log, and history.

- Frontend: minimal React app (clean light theme, no animations) in `frontend/`
- Backend: FastAPI in `backend/app.py`, wrapping the existing validation,
  retrieval, AI workflow, and SQLite storage

---

## Quick start: install, run, test

These steps work on a fresh clone with no prior setup.

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node.js 20+ and npm

### 1. Clone

```bash
git clone https://github.com/arthavjain103/marketplace.git
cd marketplace
```

### 2. Install the backend

```bash
uv sync --dev
```

Installs the app plus dev tools (pytest, ruff, pytest-cov, httpx) so you can run
the test suite and linter right away.

### 3. Configure the environment

```bash
cp .env.example .env
```

Open `.env` and fill in real values:

| Variable | Needed for | Notes |
|---|---|---|
| `PINECONE_API_KEY` | AI review, ingestion | Pinecone console |
| `PINECONE_INDEX` | AI review, ingestion | Your Pinecone index name |
| `GROQ_API_KEY` | AI review | Groq console |
| `LOG_LEVEL` | optional | Defaults to `INFO` |

`.env` is gitignored — never commit it. Only `.env.example` is tracked.
Deterministic checks and the test suite work without API keys; AI review and
ingestion need them.

### 4. Install the frontend

```bash
cd frontend
npm install
cd ..
```

### 5. Ingest the policy PDFs (Pinecone + BM25)

```bash
python ingest.py
```

Reads `docs2/*.pdf`, structure-aware chunks them, uploads vectors to Pinecone,
and saves the BM25 model to `data/bm25.pkl`. Required for AI review and
retrieval.

### 6. Run

Backend (terminal 1):

```bash
uvicorn backend.app:app --port 8000
```

Frontend (terminal 2):

```bash
cd frontend
npm run dev   # http://localhost:5173, proxies /api to :8000
```

Open http://localhost:5173. In the header, **Initialize models**, enter a
**Reviewer name** (must differ from the seller; self-approval is blocked), then
use the Review, Batch, Compare, History, or Ask tabs.

CLI still available: `python main.py` (1 = policy question, 2 = listing review;
needs ingested data).

### 7. Test

```bash
uv run pytest tests/ -q                    # 19 tests, no API keys required
uv run pytest tests/ --cov                 # with coverage (min 50%)
```

- `tests/test_validator.py` — deterministic checks
- `tests/test_store.py` — review/finding flow, append-only audit log
- `tests/test_backend.py` — API guards, outcome ladder, revised-text builder

### 8. Lint

```bash
uv run ruff check .            # lint only
uv run ruff format --check .   # format check
uv run ruff check --fix .      # auto-fix safe issues
uv run ruff format .           # format code
```

Configuration is in `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.format]`.

### 9. Docker one-command startup

Alternatively, start the whole app (backend + frontend) with a single command:

```bash
cp .env.example .env    # fill PINECONE_API_KEY, PINECONE_INDEX, GROQ_API_KEY
docker compose up       # backend on :8000, frontend on :5173
```

Docker Compose automatically loads `.env`, so fill in your Pinecone and Groq
keys there for AI review and ingestion. Deterministic checks and history work
without them.

- `backend` (port 8000): FastAPI API; runs `ingest.py` on first start to build
  `data/bm25.pkl` from `docs2/*.pdf`, then serves the API.
- `frontend` (port 5173): React + Vite dev server; proxies `/api` to the
  backend via `VITE_PROXY_TARGET=http://backend:8000`.
- `marketplace-data` volume: persists SQLite (`data/reviews.db`) and the BM25
  model (`data/bm25.pkl`) across restarts.

Open http://localhost:5173. Then, in the UI header, **Initialize models**,
enter a **Reviewer name**, and use the Review / Batch / Compare / History / Ask
tabs.

Stop with `docker compose down`. Data persists in the `marketplace-data`
volume; remove it with `docker compose down -v` to start fresh.

---

## 1. Approach and design decisions

The core idea: **never trust the model for anything a rule can decide, and never
let the model assert a rule it was not given.** Everything below follows from that.

### 1.1 Why deterministic checks and AI review are separate

| Layer | Decides | Why |
|---|---|---|
| `validator.py` (no AI) | required fields, price format/range, title & description length, supported category, category attributes, condition enum, tag count/length, contact details, exact + near duplicates, keyword stuffing, caps/emoji | These are objective, rule-bound and must be reproducible. An LLM would add cost, latency and non-determinism for zero accuracy gain |-|
| `llm/chat.py` (AI) | unclear / misleading / prohibited / incomplete wording, unverifiable claims, assumptions, severity, suggested rewording | These need judgement and language understanding; they cannot be expressed as a regex |-|

Both run on every review, and the UI renders them in **separate sections**. The
deterministic result is returned even when the LLM fails (`result["error"]`), so a
Groq outage degrades the tool instead of breaking it.

### 1.2 Chunking strategy: policy PDF -> chunks

Policy PDFs are rule-structured, so fixed-size chunking destroys the thing the
model needs (a whole rule with its ID). `ingestion/chunker.py` therefore does
**structure-aware, rule-based chunking**.

**Step 1 — merge pages per file** (`ingestion/pdf_loader.py`)
All pages of one PDF become a single `Document`. A rule routinely spans a page
boundary, so page-level splitting would cut rules in half.

**Step 2 — route by document type** (`chunk_documents`)
- policy / brand guide (`is_policy_doc`: >= 10 `POL-`/`BRD-` IDs found) -> structural chunking
- email PDFs -> template chunking (legacy path, kept)
- anything else -> `RecursiveCharacterTextSplitter` (700 chars, 150 overlap)

**Step 3 — structural chunking** (`split_policy_document`)
- Skip cover + table of contents: the body starts at the **last** `Part A. Policy` line.
- Split the body on section headings, then chunk inside each section.
- **One rule = one chunk.** Content is `[9. Prohibited Claims | POL-6.2]\nPOL-6.2: ...`,
  so the rule ID is inside the chunk text *and* in metadata. This is what makes
  keyword search for `POL-6.2` work later.
- **Zero overlap for rules.** A rule is self-contained; overlap would duplicate text
  and inflate BM25/rerank scores for the same rule. Overlap (150) is used only in the
  generic fallback splitter, where continuity matters.
- Bulleted rules (e.g. POL-4.1 category list) additionally emit one
  `category_requirements` chunk per bullet with a `category` metadata key, so
  retrieval can fetch "the attribute rules for *Electronics*" directly.
- Rewrite examples (`Problem:` / `Suggested rewrite:`) and appendix checks
  (`Pass condition:`) become one chunk per example/check — they are self-contained
  and useful few-shot style guidance.
- **Stable `chunk_id`** (`POL-6.2`, `POL-4.1-cat-3`, `intro-15`, ...) makes
  re-ingestion idempotent in Pinecone and gives the dedupe key during retrieval.

Result: chunk boundaries align with policy meaning, not with character counts.

### 1.3 Embedding choice

`intfloat/multilingual-e5-base` on CPU, embeddings normalised, with the required
`passage:` / `query:` prefixes (`ingestion/embedder.py`). E5 without these prefixes
loses a large amount of retrieval quality. Same class is used on the search side, so
index and query stay consistent. Normalised vectors make Pinecone cosine scores
comparable across queries.

Pinecone metadata is scrubbed (no `None`, no empty values) and ids are
`source::chunk_id`, so re-running `ingest.py` overwrites instead of duplicating.

### 1.4 Retrieval strategy

**Hybrid BM25 + vector, fused with RRF, then reranked** (`pro/hybrid.py`).

Why hybrid: the two retrievers fail in opposite directions.
- BM25 (`pro/keyword_based.py`) is exact and cheap — it finds `POL-6.2` or
  "prohibited claims" verbatim, but misses paraphrases.
- Pinecone vectors (`pro/semantic.py`) catch paraphrases, but can drift on exact
  identifiers and rare tokens.

Fusion uses **Reciprocal Rank Fusion (k = 60)** rather than score blending: BM25
scores are unbounded and cosine scores are 0–1, so raw scores are not comparable;
RRF only uses *ranks*, which needs no normalisation and has no tuning knob.

Pipeline: fetch `top_n * 5` from each retriever -> RRF fuse -> keep top 25 ->
**cross-encoder rerank** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) -> return `top_n`.
A cheap bi-encoder recalls; an expensive cross-encoder reorders. Metadata rides along
on every candidate so the rule ID survives reranking.

**Listing-specific retrieval** (`retrieve_for_listing` in `llm/chat.py`) is three
deliberate sources, not one generic query:

1. **Failed-check rules, by direct lookup.** Every rule ID from a failed deterministic
   check is looked up in the BM25 store by `chunk_id` — no search needed. This
   *guarantees* the exact violated rule is in context.
2. **Category requirements chunk.** The `category_requirements` chunk for the
   listing's category is fetched, so required attributes for that category are always
   present.
3. **Hybrid search with query expansion.** Query 1 is `category + title + description[:300]`.
   Then up to 4 trigger phrases become `"<phrase> claim rules"` queries. Trigger-driven
   expansion is what pulls in the prohibited-claims/health-claim rules for a listing
   containing "guaranteed" or "cure".

Ordering, dedupe and limits: dedupe by `chunk_id`; search results first (best-first
reranker order is preserved), fixed lookups appended so they are never crowded out;
each chunk trimmed to 600 chars (trim, never drop — a dropped rule is a missing
citation); final context capped at 10,000 chars.

### 1.5 Trigger phrases are signals, not failures

`MODEL_TRIGGER_PHRASES` ("guaranteed", "cure", "doctor", "100%", "best", ...) only
decide **where the AI should look**. They never fail a check on their own, because
"doctor recommended" is legitimate in some categories and "best" in others. This keeps
deterministic output defensible and leaves judgement to the AI.

### 1.6 Grounding the AI (the most important safeguard)

The model is only allowed to cite rules it was actually shown.

- Prompt forbids invented rule IDs; temperature is `0` for reproducibility.
- `validate_review()` intersects every cited ID with the IDs present in the ingested
  data. **A finding with no surviving rule ID is not a policy finding** — it is moved
  to `style_suggestions`, so a hallucinated citation can never become a finding
  (POL-10.3).
- `quote` must be copied from the listing; suggested wording may not invent facts and
  must use placeholders (`[add battery health %]`) when a fact is missing.
- Unverifiable claims must be labelled unverifiable, never "corrected" with invented
  evidence.
- Severities are normalised to Critical/Major/Minor/Info; findings are sorted by
  severity; `issue_type` and `field` are whitelisted.
- `revised_listing` is restricted to `title`, `description`, `tags`.

### 1.7 Making LLM output safe to consume

- JSON mode is requested; if the model rejects `response_format`, `_ask` transparently
  retries in normal mode.
- `_parse_json()` strips code fences, extracts the first balanced `{...}` block
  (string/escape aware), and repairs trailing commas before giving up.
- **Max 2 attempts.** The second attempt re-sends the broken reply and asks for a
  corrected JSON object only. Beyond that the review degrades gracefully: the
  deterministic result is still returned with `error` set, never an exception.

### 1.8 Security and operational safeguards

- Listing values are wrapped in explicit untrusted-data delimiters before they are
  sent to the model. Instruction-like phrases such as "ignore previous" and
  "system prompt" are deterministic findings under `POL-10.3`, not model commands.
- Reviewer identity comparison collapses whitespace and uses Unicode-aware
  case-folding, so case and spacing variants cannot bypass self-review protection.
- Request models enforce field lengths, tag counts, batch size, and retrieval limits
  server-side. These limits do not depend on the frontend.
- The `decisions` table has database-level update/delete triggers and a SHA-256
  hash chain. Verify it with `GET /api/audit/verify`.
- Groq failures use bounded retries and a circuit breaker; deterministic checks are
  still returned when the AI path is unavailable. Successful review payloads are
  cached by listing, retrieved policy context, and retrieval settings.

### 1.9 Severity and outcome

Severity ladder: Critical (illegal/unsafe/fraudulent, medical claims) > Major
(misleading/incomplete) > Minor (style) > Info (note/assumption).

A finding is **resolved** when the reviewer approves or edits it. Rejecting — or
simply not deciding — leaves it open.

- **Rejected** = a Critical finding is still open
- **Needs changes** = a Major finding is still open (a rejected Major fix keeps it open)
- **Approved** = no Critical or Major left open

Deterministic failures are stored as Major.

### 1.10 Storage decisions

SQLite (`store.py`), four tables plus one:

- `listings`, `reviews`, `findings`, `decisions` — and `ai_revisions`, which stores the
  model's `revised_listing` per review so Compare can rebuild the revised text on any
  later request (without it, "approve as-is" silently fell back to the original).
- `rule_ids` is stored as JSON and **never dropped**; it survives UI, DB and history.
- `decisions` is **append-only**: `log_decision` validates the action and requires a
  non-empty reviewer, and no update/delete path exists for that table. Reviews are
  mutable (outcome changes); the audit log is not.
- **Self-approval is blocked** server-side: reviewer must differ from the seller.
- Every DB call is serialized with a lock and commits per write.

### 1.11 UI decision: Streamlit -> React

The review logic, validation, retrieval, prompts and storage were kept byte-for-byte
and only the presentation layer was replaced. `backend/app.py` is a thin FastAPI
adapter over the same functions (`run_all_checks`, `review_listing_with_ai`,
`store.*`), so the browser never owns business logic.

- React + Vite, plain CSS, light theme, no animation libraries — deliberately minimal.
- Five tabs only: Review, Batch, Compare, History, Ask.
- Vite proxies `/api` to `127.0.0.1:8000`, so the browser sees one origin and CORS
  never bites in development.
- Batch CSV/JSON is parsed **client-side**, which removed the `pandas` dependency from
  the server and keeps the API JSON-only.
- The category dropdown falls back to a bundled list if `/api/categories` is
  unreachable, so the form stays usable while the backend is down.
- Compare refetches the review whenever the decision set changes (not only when the
  outcome changes), otherwise approving a Minor finding would not refresh the view.

### 1.12 Observability

`app_logging.py` emits **structured JSON logs** (one object per line, `event`, `data`,
`exc`). Application events (`models_initialized`, `review_created`, `decision_recorded`)
and AI-workflow events (`review_start`, `review_attempt_failed`, `review_done`,
`review_failed`, `qa_answer`) carry counts and timings, so a review can be traced
from retrieval through to the final finding count without reading code.

---

## 2. Listing outcome

- **Approved** = no Critical or Major finding left open
- **Needs changes** = a Major finding is still open (including a rejected Major fix)
- **Rejected** = a Critical finding is still open

---

## 3. Architecture

```
frontend/            React + Vite, plain CSS light theme
  Review | Batch | Compare | History | Ask
        -> REST /api/*  (Vite dev proxy, same origin)
backend/app.py       FastAPI adapter + endpoints, no business logic
  deterministic   -> validator.run_all_checks / find_model_triggers
  retrieval       -> pro/ (BM25 + Pinecone, RRF fusion, cross-encoder rerank)
  AI              -> llm/chat.py (Groq, JSON repair + retry, rule-ID grounding)
  storage         -> store.py (SQLite: listings, reviews, findings, decisions, ai_revisions)
ingestion/          pdf_loader -> chunker -> embedder -> pinecone_upload
ingest.py           one-shot ingestion entry point
app_logging.py      structured JSON logging
tests/              pytest suite
```

---

## 4. Completed scope

- Listing fields: title, description, category, price, attributes, seller, optional tags
- AI workflow: retrieve policy + brand guidance, detect unclear/misleading/prohibited/
  incomplete content, cite a policy section for every finding, suggest improved
  wording, surface assumptions and unverifiable claims, classify severity
- Deterministic validation: required fields, price format, duplicate detection,
  title/description length, supported categories
- User actions: field-by-field approve/edit/reject, original vs revised comparison,
  small-batch processing, preserved review and approval history
- Structured application and AI-workflow logs; focused tests

---

## 5. Excluded scope (per assignment)

Publishing to a real marketplace, image moderation, payments, seller verification,
unrestricted product categories.

---

## 6. Limitations

- AI review needs `GROQ_API_KEY` and a populated Pinecone index; deterministic checks
  and history work without them once `data/bm25.pkl` exists.
- Retrieval quality is bounded by the ingested policy: a rule missing from the PDF
  cannot be cited.
- Findings are AI-generated and grounded, not verified truth — they inform a human
  reviewer, they do not replace one.
- SQLite and a single-process in-memory model cache suit single-user review; use a
  server database and a shared model cache for multi-user deployment.
- Duplicate detection is same-seller scoped (normalised exact title, or >= 90%
  similarity on title/description within a category).
- Batch processing is capped at 5–10 listings by design.

---

## 7. Deployment

- Backend: `uvicorn backend.app:app --host 0.0.0.0 --port 8000`, env supplied by the
  host (`.env` is loaded locally via `python-dotenv`)
- Frontend: `cd frontend && npm run build`, serve `frontend/dist/` from any static host
  that proxies `/api` to the backend
- Never commit `.env`; only `.env.example` is tracked