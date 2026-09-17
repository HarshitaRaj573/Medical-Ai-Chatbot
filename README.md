<<<<<<< HEAD
# Medical-Ai-Chatbot
=======
# Medical AI Chatbot

This beginner-friendly chatbot answers medical education questions using PDF
documents and MedQuAD-style CSV data.

> **Important:** This project is educational. It does not replace a doctor,
> diagnose illness, or manage emergencies. Do not add private patient data.
> Embeddings run locally, but the retrieved context and question are sent to
> Gemini when the chatbot creates its final answer.

## 1. What this project does

The chatbot first searches your medical data and then asks Gemini to answer from
the retrieved information. This is called **RAG** (retrieval-augmented
generation).

An **embedding** is a list of numbers representing a text's meaning. The free
`sentence-transformers/all-MiniLM-L6-v2` model creates 384-number embeddings on
your computer. Pinecone stores them so similar medical text can be found.

The flow is:

1. `ingest.py` reads PDF pages and CSV question-and-answer rows.
2. LangChain splits PDF text and only unusually large CSV answers.
3. MiniLM creates embeddings locally, without Gemini quota.
4. Pinecone stores the embeddings and source metadata.
5. `app.py` embeds a question locally and retrieves five relevant records.
6. Gemini writes an answer from that context and the app shows its sources.

## 2. Requirements

You need:

- Python 3.10 or newer
- A Google AI Studio account and Gemini API key
- A Pinecone account and API key
- Medical PDF or CSV data you are allowed to use

Gemini and Pinecone API use may be subject to provider limits or costs. Local
embedding does not consume Gemini API quota.

## 3. Create a virtual environment

Open a terminal in this project folder.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate
```

## 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The first local embedding run downloads the MiniLM model (roughly 90 MB). It is
then reused from your computer's cache.

## 5. Create `.env`

Copy `.env.example` to `.env`, then add your real keys:

```text
GEMINI_API_KEY=your_real_gemini_key
PINECONE_API_KEY=your_real_pinecone_key
PINECONE_INDEX_NAME=medical-chatbot-local
```

Never share or commit `.env`. Gemini is required when chatting, but it is not
used by `ingest.py`.

## 6. Pinecone

Pinecone stores the searchable medical knowledge base. `ingest.py` creates
`medical-chatbot-local` automatically with:

- 384 dimensions
- cosine similarity
- AWS `us-east-1` serverless configuration

The older 1,536-dimensional index cannot store these 384-dimensional vectors.
This project uses the new index and does not delete the old one.

## 7. Add PDFs and CSV data

Put data directly inside `data/`:

```text
data/
  medical_book.pdf
  medDataset_processed.csv
```

CSV files must contain `Question` and `Answer` columns. The optional `qtype`
column is saved as question-type metadata. The current MedQuAD CSV contains
`qtype`, `Question`, and `Answer`.

Normal Q&A rows remain together. Very large answers may be split into a few
parts. PDFs are split into overlapping chunks and retain filename/page metadata.
Scanned PDFs need OCR so their text is selectable.

## 8. Index the data

```bash
python ingest.py
```

The script streams CSV records in batches of 250, creates embeddings locally,
and uploads them to Pinecone. It can take time for a large PDF and CSV. Stable
IDs make identical records overwrite themselves on another run instead of
creating duplicates.

The denominator in progress messages says “at least” because an unusually long
CSV row can become more than one record.

## 9. Run the chatbot

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, often `http://localhost:8501`.
Stop the app with `Ctrl+C`.

## 10. Typical workflow

1. Activate `.venv`.
2. Add a PDF or MedQuAD CSV to `data/`.
3. Run `python ingest.py`.
4. Run `streamlit run app.py`.
5. Ask medical education questions and review the displayed sources.

## 11. Common errors

### `GEMINI_API_KEY is missing from .env.`

Add your Google AI Studio key after `GEMINI_API_KEY=`. This key is checked only
when Gemini needs to write a chat answer.

### `PINECONE_API_KEY is missing from .env.`

Add your Pinecone key after `PINECONE_API_KEY=` and save `.env`.

### Existing index has the wrong dimension

Set `PINECONE_INDEX_NAME=medical-chatbot-local` in `.env`. The program will
create the new 384-dimensional index rather than change or delete the old one.

### CSV cannot be read

Confirm it is a valid UTF-8 CSV containing `Question` and `Answer` headers. The
optional `qtype` header may be absent.

### PDF contains no readable text

The PDF may be a scanned image. Use OCR or try a text-based PDF. Unlock
password-protected files first.

### Knowledge base appears empty

Run `python ingest.py` successfully. Confirm both scripts use the same
`PINECONE_INDEX_NAME` from `.env`.

### Local model download fails

Check your internet connection and rerun the command. The model only needs to
download once and is cached afterward.

### Gemini quota or rate limit reached

Wait briefly and try the chat again, or check the API project's usage limits.
This does not affect local ingestion.

### `ModuleNotFoundError`

Activate `.venv` and run `python -m pip install -r requirements.txt` again.

## Project files

- `ingest.py`: reads PDFs/CSVs and uploads locally embedded records in batches
- `app.py`: retrieves locally and asks Gemini to create the final answer
- `data/`: holds PDFs and CSV files
- `.env.example`: safe settings template
- `requirements.txt`: required Python packages

This simple version intentionally has no accounts, SQL, Redis, agents, Docker,
FastAPI, React, prescriptions, or admin dashboard.
>>>>>>> 4a50718 (Build medical AI chatbot with MedQuAD, Pinecone and Gemini)
