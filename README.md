# Paper Search and Analysis System

The system is a Flask-based academic paper search system for a closed paper corpus. It combines hybrid dense/BM25 retrieval, an optional PASA/Qwen selector reranker, DeepSeek-powered query rewriting and analysis, PDF management, and a shared RAG evaluation pipeline for dense, sparse, and hybrid retrieval experiments.

The web application is designed for interactive paper discovery: users can search in natural language, receive real-time results through Socket.IO, inspect paper metadata, generate BibTeX, upload and review papers, view PDFs, and ask citation-oriented questions about selected papers.

## Features

- Hybrid paper search with `all-MiniLM-L6-v2` dense retrieval, BM25 sparse retrieval, and Reciprocal Rank Fusion.
- Optional local selector reranking with `model/selector`.
- Real-time search result streaming through Flask-SocketIO.
- Chinese query translation and English spelling correction before retrieval.
- DeepSeek API integration for query rewriting, paper analysis, citation suggestions, and paper chat.
- PDF upload, preview, download, and text extraction.
- Tencent COS support for cloud PDF storage, with local storage fallback.
- Admin pages for pending paper review, paper updates, deletion, duplicate checks, and search statistics.
- RAG utilities for Milvus/in-memory dense retrieval, BM25 sparse retrieval, hybrid Reciprocal Rank Fusion, and evaluation reports.

## Repository Layout

```text
.
|-- app.py                         # Main Flask + Socket.IO web application
|-- config/
|   `-- config.py                  # Runtime paths, model paths, API keys, COS, and RAG settings
|-- coding/
|   |-- paper_search.py            # Main hybrid search, query rewriting, rerank, BibTeX, analysis
|   |-- models.py                  # Local selector model wrapper
|   |-- cos_utils.py               # Tencent COS upload/download/presigned URL helpers
|   |-- db_models.py               # SQLAlchemy model definitions
|   |-- setup_cos.py               # Interactive COS setup helper
|   |-- evaluation/                # Selector and metric plotting utilities
|   `-- rag/
|       |-- paper_rag/             # RAG loader, dense store, BM25, fusion, reranker, evaluation
|       `-- scripts/
|           |-- build_rag_index.py     # Build dense/BM25 retrieval index from papers.db
|           |-- test_preretrieval.py   # Run a single retrieval query from the command line
|           `-- evaluate_retrieval.py  # Evaluate dense/sparse/hybrid retrieval and rerank
|-- src/
|   |-- papers.db                  # SQLite paper database used by default
|   |-- agent_prompt.json          # Prompt templates for selector/query tasks
|   |-- templates/                 # Web UI templates
|   `-- frequency_dictionary_en_82_765.txt
|-- dataset/
|   `-- testset.jsonl              # Retrieval evaluation set
|-- model/
|   |-- all-MiniLM-L6-v2/          # Local sentence-transformer model
|   `-- pasa-7b-selector/          # Local selector/reranker model
`-- output/eval/                   # Saved evaluation reports
```

## Requirements

- Python 3.10+ recommended.
- CUDA-capable GPU strongly recommended for the selector model.
- Enough disk space for local model files.
- A populated SQLite database with a `papers` table.
- DeepSeek API key for query rewriting, translation, chat, and citation analysis.
- Tencent COS credentials only if `USE_COS_STORAGE=true`.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Optional RAG dense indexing with Milvus Lite requires `pymilvus`, which is used by `coding/rag/paper_rag/milvus_store.py` but is not currently listed in `requirements.txt`:

```bash
pip install pymilvus
```

If Milvus is unavailable and `RAG_ENABLE_IN_MEMORY_FALLBACK=true`, the RAG pipeline falls back to an in-memory vector store.

## Configuration

Most settings live in `config/config.py` and can be overridden with environment variables.

Important variables:

```bash
# DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# Model paths
export MODEL_PATH="model/all-MiniLM-L6-v2"
export SELECTOR_PATH="model/selector"

# Data paths
export DATABASE_PATH="src/papers.db"
export TESTSET_PATH="dataset/testset.jsonl"

# Tencent COS, only required when cloud PDF storage is enabled
export USE_COS_STORAGE="false"
export COS_SECRET_ID="your-secret-id"
export COS_SECRET_KEY="your-secret-key"
export COS_REGION="ap-guangzhou"
export COS_BUCKET_NAME="your-bucket"

# RAG
export RAG_ENABLE_MILVUS="true"
export RAG_ENABLE_IN_MEMORY_FALLBACK="true"
export RAG_MILVUS_URI="output/rag/milvus.db"
export RAG_MILVUS_COLLECTION="pasa_papers"
```

Security note: do not rely on credentials committed in source files for production. Override them with environment variables and rotate any exposed keys before deployment.

## Database

The default database path is selected from these locations, in order of availability:

1. `src/papers.db`
2. `src/resources/papers.db`
3. `/root/autodl-fs/pasa/src/resources/papers.db`
4. `/autodl-fs/data/pasa/src/resources/papers.db`

`app.py` creates or migrates the `papers` table on startup. Core fields include:

| Field | Purpose |
| --- | --- |
| `id` | Paper identifier |
| `title` | Paper title |
| `authors` | Author list |
| `abstract` | Abstract text |
| `year` | Publication year |
| `journal` | Journal or venue |
| `doi` | DOI |
| `status` | Review status, such as `pending` or `approved` |
| `submitter` | User who submitted the paper |
| `review_comment` | Admin review comment |
| `reviewed_by` | Reviewer identifier |
| `submitted_at` | Submission timestamp |
| `reviewed_at` | Review timestamp |
| `pdf_file_path` | Local path or COS key for the PDF |

The RAG loader currently retrieves every row in `papers` ordered by `id` and builds one retrieval chunk per paper from title, authors, year, venue, keywords, and abstract.

## Run the Web App

Start the Flask-SocketIO server:

```bash
python app.py
```

By default the app listens on:

```text
http://0.0.0.0:6006
```

Main pages:

- `/` - search interface
- `/upload` - paper/PDF submission
- `/admin` - admin dashboard
- `/admin/review` - pending paper review

Useful API endpoints include `/search`, `/search_realtime`, `/paper_chat`, `/citation_chat`, `/view_pdf/<paper_id>`, `/download_pdf/<paper_id>`, and `/admin/search_stats`.

## Search Flow

1. `PaperSearch` loads the local sentence-transformer model.
2. Paper records are loaded from SQLite.
3. Title and abstract text are embedded for dense semantic search, and a BM25 sparse index is built from the same paper corpus.
4. User queries are normalized:
   - Chinese text can be translated to English through DeepSeek.
   - English spelling can be corrected with SymSpell.
5. The online frontend requests `retrieval_mode: hybrid` by default.
6. Dense and BM25 candidates are fused with Reciprocal Rank Fusion to produce the top retrieval candidates.
7. If the selector model is available, candidates are reranked and filtered by selector score.
8. Results are returned with metadata, hybrid retrieval score, dense/BM25 branch scores, selector score, BibTeX, and optional PDF paths.

The web app also supports multi-level query generation. DeepSeek rewrites an input query into broad, moderate, and specific research queries, then the selected level is searched.

## RAG Indexing and Retrieval

Build or refresh the RAG index:

```bash
python coding/rag/scripts/build_rag_index.py --force
```

Run a single retrieval query:

```bash
python coding/rag/scripts/test_preretrieval.py "graph neural networks for traffic prediction" --mode hybrid --top-k 10
```

Supported retrieval modes:

- `dense` - sentence-transformer embeddings with Milvus or in-memory vector search
- `sparse` - in-memory BM25
- `hybrid` - Reciprocal Rank Fusion over dense and sparse results

## Evaluation

The evaluation set is a JSONL file where each row contains a question and one or more gold paper IDs:

```json
{
  "qid": "PaperQuery_0",
  "question": "What convergence analyses exist for Oja's PCA and MCA learning algorithms?",
  "answer": ["Convergence analysis of a deterministic discrete time system of Oja's PCA learning algorithm"],
  "answer_paper_id": [883]
}
```

Evaluate retrieval:

```bash
python coding/rag/scripts/evaluate_retrieval.py --retrieval-mode all --candidate-k 50 --final-k 5
```

Evaluate retrieval plus selector reranking:

```bash
python coding/rag/scripts/evaluate_retrieval.py --retrieval-mode hybrid --candidate-k 50 --final-k 5 --rerank
```

Outputs are written to `output/eval/`:

- `*_summary.json`
- `*_details.json`
- `*_details.csv`
- `*_metrics.csv`
- `*_report_metrics.csv`

Metrics include hit/retrieval quality at configured K values, final output metrics, mean latency, P50 latency, and P95 latency.

## PDF Storage

PASA supports two PDF storage modes:

- Local mode: set `USE_COS_STORAGE=false`; uploaded PDFs are stored under `src/pdfs` by default.
- Tencent COS mode: set `USE_COS_STORAGE=true` and configure COS credentials, region, bucket, and folder.

PDF text is extracted with `PyPDF2` for paper chat and analysis features. COS files are downloaded into memory before extraction.

## Notes and Limitations

- Several source comments and older README text appear to have been saved with the wrong character encoding, but the runtime logic is still readable from function names and code structure.
- The local selector model can require significant GPU memory. If you only need embedding search, set `SELECTOR_PATH` to an unavailable path or adapt the app to initialize `PaperSearch` without a selector.
- `pymilvus` is optional but required for the Milvus backend.
- `python.sts.sts` is imported by one COS STS helper path and may require Tencent's STS package if that endpoint is used.
- The app initializes models at startup, so first launch can be slow.

## License

The selector wrapper in `coding/models.py` includes an Apache-2.0 header from ByteDance. Check the licenses of bundled model directories before redistribution or deployment.
