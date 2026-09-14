# Semantic Corpus Search

Query-by-example document retrieval: given any reference document, rank an entire corpus of
unstructured documents by semantic similarity. Built to screen 100+ heterogeneous documents
against a reference in one pass.

This is a dense-retrieval / semantic-search pipeline, not a classifier. It answers *"which
documents are most related to this one?"* and leaves judgment to you.

## What it does

- Ingests mixed corpora: PDF, .pages, .docx, and plain text
- Embeds every document with a sentence-transformer model
- Scores the corpus against a reference via a vectorized cosine-similarity matrix
- Returns top-k matches above a tunable threshold, plus a score-distribution summary
- Persists results to JSON for downstream tooling

## Pipeline

```
parse ──> embed ──> score ──> rank ──> report
 PDFs        all-mpnet-     vectorized      threshold,     JSON +
 Pages       base-v2        cosine matrix   top-k          distribution stats
 DOCX
```

Three stages, each a discrete module:

| Stage | Module | Responsibility |
|---|---|---|
| Parse | `document_parser.py` | Extract text from PDF/.pages/.docx into a doc-id → text map |
| Embed | `embedding_agent.py` | Batch dense embeddings via `SentenceTransformer` (model-agnostic — swap any model) |
| Search | `similarity_search.py` | Query-by-example scoring, cosine matrix, threshold/top-k, score distributions |

## Why brute-force cosine and not FAISS

For medium corpora (10²–10³ documents) a full vectorized cosine matrix is O(n²) but trivially
fast and exact — no index quantization, no recall loss, simpler to debug. For larger corpora the
search agent could be swapped for a FAISS index with no changes elsewhere in the pipeline.

## Usage

```bash
./setup.sh                         # create venv + install deps
python main.py \
  --reference /path/to/reference.pdf \
  --search-dir  /path/to/corpus/ \
  --output      results.json \
  --threshold   0.3 \
  --top-k       50
```

Output writes to `results.json`:

```json
{
  "threshold": 0.3,
  "matches": [
    { "document": "/path/to/doc_1.pdf", "similarity": 0.61 },
    { "document": "/path/to/doc_2.pdf", "similarity": 0.55 }
  ]
}
```

## Notes

- Similarity is a *signal*, not a verdict — scores vary by model and domain. Tune `threshold`
  per corpus using the distribution stats.
- The reference document is itself embedded as part of the corpus, so you can also map
  *clustering* structure across the whole set, not just ties to the reference.
- Full batch embedding keeps large corpora memory-friendly (32-doc batches by default).