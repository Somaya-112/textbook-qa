# Textbook helper

Ask questions about a PDF textbook and get answers **grounded in the book, with page citations**. If the book doesn't cover the question, it says so instead of guessing.

Built to run on a low-end laptop (2 GB GPU) using free LLM APIs.

## Features

- **Cited answers:** every answer links to the source pages. Clicking a citation jumps the PDF viewer to that page.
- **"Not in the book" detection:** a relevance threshold on reranker scores skips the LLM when retrieval is weak, so it doesn't hallucinate.
- **Study modes:** explain simply, exact definition, or quiz me.
- **Follow-up questions:** chat history is used to rewrite vague questions like "what about its cause?" into standalone ones.
- **Student-friendly UI:** split PDF and chat view, mobile tabs, a mascot, levels, read-aloud, and reduced-motion support.
- **Reliability:** LLM fallback chain (Groq, then a backup key, then Gemini), answer caching, and per-user rate limiting.

## How it works

```
PDF -> PyMuPDF (text + table of contents) -> chapter-aware chunks (with page ranges)
    -> local GPU embeddings (bge-small) -> Qdrant (local, on-disk)

Question -> vector search (filtered to one book) -> FlashRank reranker
         -> relevance threshold -> LLM with a strict "context only" prompt
         -> answer + [p. N] citations
```

## Tech stack

Python, FastAPI, PyMuPDF, sentence-transformers, FlashRank, Qdrant (local mode), Groq and Gemini via the OpenAI-compatible API, plain HTML/CSS/JS frontend.

## Run it

```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
# install PyTorch for your system first: https://pytorch.org
pip install -r requirements.txt
cp .env.example .env             # then add your Groq / Gemini keys
python -m uvicorn main:app --port 8000
```

Open http://localhost:8000, click **Add a book**, and ask a question.

## Design decisions

- **Plain Python instead of an agent framework:** the flow is mostly linear, so it is easier to read and debug.
- **Chunking follows the book's structure:** chunks never cross chapters and keep their page range, which is what makes accurate citations possible.
- **Retrieval is always filtered by book:** answers can't leak between books.
- **Retrieved text is treated as data, not instructions:** a basic defense against prompt injection.

