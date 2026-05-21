---
name: building-testset
description: Build a JSONL evaluation testset for the Paper Search and Analysis System. Use when the user wants to create, expand, or curate a retrieval benchmark in the format {question, answer, answer_paper_id, source_meta, qid}, where paper identifiers come from the project's own SQLite database (Paper.id) rather than external IDs like arxiv IDs. Triggers on requests like "build a testset", "create evaluation data", "generate benchmark queries", or "add test questions to testset".
---

# Building Testset

## Output Format

Each line in the output JSONL is one query record:

```json
{
  "question": "Natural-language research query",
  "answer": ["Paper Title A", "Paper Title B"],
  "answer_paper_id": [3, 17],
  "source_meta": {
    "db_snapshot_time": "<ISO-8601 timestamp fetched at generation time>",
    "db_paper_count": 152,
    "collection": "paper_submissions"
  },
  "qid": "PaperQuery_0"
}
```

### Field Rules

| Field | Rule |
|-------|------|
| `question` | Natural-language research query, written or validated by a human |
| `answer` | Paper titles in the **same order** as `answer_paper_id` |
| `answer_paper_id` | `Paper.id` (integer primary key from the local SQLite DB) — **never** arxiv IDs or DOIs |
| `source_meta` | Populated dynamically from the live DB at generation time — **never hardcode** counts or timestamps |
| `qid` | Format: `PaperQuery_<N>`, zero-indexed, unique within the file |

---

## Database Schema (relevant fields)

The project uses SQLite (`papers.db`). The two key tables:

### `paper` — approved papers (use this for testset)

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Use as `answer_paper_id` |
| `title` | TEXT | Use as `answer` |
| `abstract` | TEXT | Use for relevance matching |
| `authors` | TEXT | |
| `year` | INTEGER | |
| `journal` | TEXT | |
| `doi` | TEXT | Do NOT use as identifier in testset |
| `keywords` | TEXT | |
| `citation_key` | TEXT | |
| `booktitle` | TEXT | |
| `added_at` | DATETIME | |

### `paper_submissions` — all submissions (pending / approved / rejected)

Same structure plus `status` column. **Do not use IDs from this table** for `answer_paper_id` — only use the `paper` table (approved only).

---

## source_meta — Dynamic Population

Always query the live DB to populate `source_meta`. Example Python snippet:

```python
import sqlite3
from datetime import datetime

conn = sqlite3.connect("papers.db")
paper_count = conn.execute("SELECT COUNT(*) FROM paper").fetchone()[0]

source_meta = {
    "db_snapshot_time": datetime.utcnow().isoformat(),
    "db_paper_count": paper_count,
    "collection": "paper_submissions",
}
```

---

## Workflow

1. **Check DB state** — query `SELECT COUNT(*) FROM paper` to know how many approved papers are available
2. **Collect questions** — get questions from the user, or generate candidates and ask the user to confirm
3. **Select ground-truth papers** — for each question, search the DB by title / abstract / keywords; resolve `id` values for matched papers; confirm with the user when uncertain
4. **Assemble records** — build one JSON object per question using the format above; fetch `source_meta` once per run (same snapshot for all records in a batch)
5. **Write JSONL** — append or create the output file; when appending, read the existing max `qid` index first to avoid duplicates

---

## Rules

- `answer_paper_id` uses `Paper.id` only — never arxiv IDs, DOIs, or citation keys
- `source_meta` is always fetched live — never hardcoded
- Only include papers from the `paper` table (approved); ignore `PaperSubmission` entries that are pending or rejected
- Skip questions with zero matching papers in the DB; warn the user
- When appending to an existing testset, parse the max `PaperQuery_<N>` index first and continue from `N+1`

---

## Example Complete Record

```json
{
  "question": "Find papers that use contrastive learning to improve sentence embeddings",
  "answer": [
    "SimCSE: Simple Contrastive Learning of Sentence Embeddings",
    "Improved Baselines with Contrastive Learning on Textual Data"
  ],
  "answer_paper_id": [12, 34],
  "source_meta": {
    "db_snapshot_time": "2026-05-07T03:14:22.819432",
    "db_paper_count": 152,
    "collection": "paper_submissions"
  },
  "qid": "PaperQuery_0"
}
```
