# RAG Resume QA — FAISS + SentenceTransformers + Groq

Simple Retrieval‑Augmented Generation (RAG) demo using FastAPI:
- Upload a PDF resume, extract text, split into chunks, embed with SentenceTransformers and store embeddings in FAISS.
- Query the index to retrieve top-k chunks and call a hosted LLM (Groq) to generate concise answers with cited chunk ids.

## Features
- PDF ingestion and text extraction (PyPDF2)
- Chunking with LangChain text splitter
- Embeddings via sentence-transformers (all-MiniLM-L6-v2)
- Vector index with FAISS (faiss-cpu)
- Query endpoint that retrieves context + calls Groq LLM
- /models debug endpoint to list available Groq models

## Repo layout
- main.py — FastAPI application (process, query, models endpoints)
- uploads/ — uploaded PDFs
- faiss.index, metadata.pkl — persisted index & metadata after processing
- .env — environment variables (not committed)

## Requirements
- Python 3.10+
- Recommended packages (add to requirements.txt):
  - fastapi, uvicorn
  - faiss-cpu
  - sentence-transformers
  - langchain
  - PyPDF2
  - python-dotenv
  - requests
  - numpy
  - pickle (stdlib)

## Setup (Windows PowerShell)
1. Create & activate venv
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create `.env` in project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Do NOT commit this file. If a key is exposed, rotate it immediately.

## Run
```powershell
python -m uvicorn main:app --reload
# Open Swagger UI: http://127.0.0.1:8000/docs
```

## Endpoints
- POST /process/ — Upload a PDF file (form field `file`) to extract, chunk, embed and store in FAISS. Returns number of chunks and embedding shape.
- POST /query/?query=...&top_k=3 — Query the index. Returns retrieved chunks and the LLM answer (calls Groq).
- GET /models — Debug endpoint to list available Groq models (reads `GROQ_API_KEY`).

## Usage examples (PowerShell)
- Upload file:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/process/" -F "file=@C:\path\to\resume.pdf"
```
- Query:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/query/?query=Tell%20me%20about%20my%20resume&top_k=3"
```
- List models:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/models"
```

## Troubleshooting
- 401/403 from Groq: verify `GROQ_API_KEY` is valid, not restricted, and your account has access. Rotate exposed keys.
- 422 in docs: Swagger shows example 422 by default — that does not mean an error happened.
- 500 JSON serialization errors: cast NumPy types (e.g., `int(idx)`) before returning JSON.
- Blocking calls: network requests using `requests` are synchronous — run them in a thread executor (asyncio.run_in_executor) or use an async HTTP client for production.

## Security & production notes
- Do not store API keys in repo. Use secret managers for production.
- Restrict CORS origins (avoid `allow_origins=["*"]` with credentials).
- For concurrency/scale, load FAISS index on startup or use a shared vector DB.
- Monitor prompt size and truncate context to respect model token limits.

## License
Choose and add a license file (e.g., MIT) if desired.
