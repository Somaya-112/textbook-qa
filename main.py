"""Web API. Run with:  uvicorn main:app --port 8000"""
import logging, time, tempfile, shutil
from collections import defaultdict, deque
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import config, rag

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Textbook helper")
hits_by_ip = defaultdict(deque)

def rate_limit(request: Request):
    """Sliding window: max N questions per minute per student."""
    q, now = hits_by_ip[request.client.host], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= config.RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "You're asking very fast. Wait a few seconds and try again.")
    q.append(now)

class Ask(BaseModel):
    book_id: str
    question: str = Field(min_length=2, max_length=500)   # input validation
    mode: str = "explain"
    history: list[dict] = []

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/api/books")
def books():
    return rag.list_books()

@app.post("/api/books")
def upload(file: UploadFile = File(...)):
    # plain `def` => FastAPI runs it in a worker thread, so slow ingestion won't freeze the server
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
    path = Path(tmp.name)
    try:
        if path.stat().st_size > 100_000_000:
            raise HTTPException(400, "That file is over 100 MB.")
        if path.read_bytes()[:5] != b"%PDF-":      # check real content, not just the name
            raise HTTPException(400, "That file isn't a valid PDF.")
        book_id = rag.ingest(path, Path(file.filename).stem)
        shutil.copy(path, config.DATA / "pdfs" / f"{book_id}.pdf")
        return {"book_id": book_id}
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        path.unlink(missing_ok=True)

@app.get("/api/books/{book_id}/pdf")
def pdf(book_id: str):
    f = config.DATA / "pdfs" / f"{Path(book_id).name}.pdf"   # .name blocks path tricks like ../
    if not f.exists():
        raise HTTPException(404, "Book not found.")
    return FileResponse(f, media_type="application/pdf")

@app.post("/api/ask")
def ask(body: Ask, request: Request):
    rate_limit(request)
    if body.book_id not in rag.list_books():
        raise HTTPException(404, "Book not found.")
    try:
        return rag.ask(body.book_id, body.question, body.mode, body.history)
    except RuntimeError:
        raise HTTPException(503, "The AI service is busy right now. Please try again in a minute.")
