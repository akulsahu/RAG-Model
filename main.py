from fastapi import FastAPI, UploadFile, File, Query    
from fastapi.responses import JSONResponse
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle  
import requests
from dotenv import load_dotenv


load_dotenv()

app=FastAPI()

faiss_index = None  # FAISS index will hold embeddings
metadata_store = {}

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")


def read_root():
    return {"FastAPI is working"}


@app.get("/models")
async def list_models():
    """List available Groq models."""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return JSONResponse(status_code=500, content={"error": "GROQ_API_KEY not set"})
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
        resp.raise_for_status()
        return JSONResponse(content=resp.json())
    except requests.RequestException as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def read_pd(file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return chunks


embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings

@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    # Save file
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Extract text
    text = read_pd(file_location)

    # Chunk text
    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = embed_chunks(chunks)

    

    global faiss_index, metadata_store
    dim = embeddings.shape[1]
    if faiss_index is None:
        faiss_index = faiss.IndexFlatL2(dim)  # L2 distance
        metadata_store = {}

    start_id = faiss_index.ntotal
    faiss_index.add(embeddings)
    for i, chunk in enumerate(chunks):
        metadata_store[start_id + i] = {"text": chunk, "source": file.filename}

    faiss.write_index(faiss_index, "faiss.index")
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata_store, f)

    return JSONResponse(content={
        "filename": file.filename,
        "num_chunks": len(chunks),
        "embedding_shape": str(embeddings.shape),
        "message": "File processed and stored in FAISS"
    })

@app.post("/query/")
async def query_documents(query: str = Query(...), top_k: int = 3):
    global faiss_index, metadata_store
    if faiss_index is None:
        if os.path.exists("faiss.index") and os.path.exists("metadata.pkl"):
            faiss_index = faiss.read_index("faiss.index")
            with open("metadata.pkl", "rb") as f:
                metadata_store = pickle.load(f)
        else:
            raise RuntimeError("No FAISS index or metadata found. Upload and process a file first.")

    query_embedding = np.asarray(embedding_model.encode([query]), dtype=np.float32)
    D, I = faiss_index.search(query_embedding, top_k)

    results = []
    contexts = []
    for idx in I[0]:
        if idx in metadata_store:
            item = {"id": int(idx), "text": metadata_store[idx]["text"], "source": metadata_store[idx].get("source")}
            results.append(item)
            contexts.append(item)
    prompt = build_prompt(query, contexts)
    try:
        answer = call_groq(prompt)
    except Exception as e:
        return JSONResponse(content={
            "query": query,
            "answer": None,
            "sources": results,
            "llm_error": str(e)
        }, status_code=500)
    return JSONResponse(content={
        "query": query,
        "answer": answer,
        "sources": results
    })


def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile", 
              max_tokens: int = 512, temperature: float = 0.2):
    """
    Calls the Groq API (OpenAI-compatible) with the given prompt.
    """
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise RuntimeError("API_KEY not set in environment variables.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=body, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Extract the output text safely
    if isinstance(data, dict) and "choices" in data:
        return data["choices"][0]["message"]["content"]
    return str(data)


# Build prompt function for Groq
def build_prompt(query: str, contexts: list, max_context_chars: int = 3000) -> str:
    """
    Build a prompt for the LLM using the retrieved contexts and the user query.
    Truncates contexts to keep the prompt reasonably sized.
    """
    included = []
    total = 0
    for c in contexts:
        chunk_text = c["text"]
        # simple char-level truncation per chunk to avoid exceeding model limits
        if total + len(chunk_text) > max_context_chars:
            remaining = max_context_chars - total
            if remaining <= 0:
                break
            chunk_text = chunk_text[:remaining]
        included.append(f"[id:{c['id']}] {chunk_text}")
        total += len(chunk_text)
    combined_context = "\n\n---\n\n".join(included)
    prompt = (
        "You are a helpful assistant. Use the provided CONTEXT to answer the QUESTION. "
        "Cite context chunk ids in your answer when you refer to specific information.\n\n"
        "CONTEXT:\n" + combined_context + "\n\n"
        "QUESTION:\n" + query + "\n\n"
        "Answer concisely and list the cited chunk ids at the end."
    )
    return prompt