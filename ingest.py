"""Read medical PDFs and CSV files, then index them in Pinecone."""

import csv
import hashlib
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 250
PINECONE_BATCH_SIZE = 50
LONG_QA_LIMIT = 3000


def require_environment_variable(name: str) -> str:
    """Return an environment variable or stop with a simple error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is missing from .env.")
    return value


def load_pdf_documents() -> list[Document]:
    """Extract readable text from PDFs while keeping filename and page."""
    documents: list[Document] = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        print(f"Loading {pdf_path.name}...")
        try:
            reader = PdfReader(pdf_path)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "document_type": "pdf",
                                "filename": pdf_path.name,
                                "page": page_number,
                            },
                        )
                    )
        except Exception as error:
            raise ValueError(
                f"Could not read {pdf_path.name}. Make sure it is a valid, "
                f"unencrypted PDF. Details: {error}"
            ) from error
    return documents


def split_pdf_documents(documents: list[Document]) -> list[Document]:
    """Split long PDF pages into searchable, overlapping chunks."""
    if not documents:
        return []
    print("Splitting PDF pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def inspect_csv(csv_path: Path) -> tuple[dict[str, str], int]:
    """Validate required columns and count usable Q&A rows."""
    print(f"Loading {csv_path.name}...")
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                raise ValueError("The CSV has no header row.")

            columns = {name.strip().lower(): name for name in reader.fieldnames}
            if "question" not in columns or "answer" not in columns:
                raise ValueError("The CSV must contain Question and Answer columns.")

            row_count = sum(
                1
                for row in reader
                if (row.get(columns["question"]) or "").strip()
                and (row.get(columns["answer"]) or "").strip()
            )
    except Exception as error:
        raise ValueError(
            f"Could not read {csv_path.name}. Check that the file exists and is "
            f"a valid CSV. Details: {error}"
        ) from error

    print(f"Found {row_count:,} rows.")
    return columns, row_count


def make_medquad_documents(
    question: str, answer: str, qtype: str, row_number: int
) -> list[Document]:
    """Keep a Q&A together, splitting only unusually large answers."""
    full_text = f"Question: {question}\n\nAnswer: {answer}"
    content_hash = hashlib.sha256(
        f"{question}\n{answer}".encode("utf-8")
    ).hexdigest()
    metadata = {
        "document_type": "medquad",
        "dataset": "MedQuAD",
        "question": question,
        "answer": answer,
        "row_number": row_number,
    }
    if qtype:
        metadata["question_type"] = qtype

    if len(full_text) <= LONG_QA_LIMIT:
        metadata["vector_id"] = f"medquad_{content_hash}"
        return [Document(page_content=full_text, metadata=metadata)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    parts = splitter.split_text(full_text)
    documents = []
    for part_number, part in enumerate(parts, start=1):
        part_metadata = metadata.copy()
        part_metadata["part"] = part_number
        part_metadata["vector_id"] = f"medquad_{content_hash}_{part_number}"
        documents.append(Document(page_content=part, metadata=part_metadata))
    return documents


def iter_csv_batches(
    csv_path: Path, columns: dict[str, str]
) -> Iterator[list[Document]]:
    """Yield CSV documents in batches instead of loading the whole file."""
    batch: list[Document] = []
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row_number, row in enumerate(reader, start=2):
                question = (row.get(columns["question"]) or "").strip()
                answer = (row.get(columns["answer"]) or "").strip()
                if not question or not answer:
                    continue
                qtype_column = columns.get("qtype")
                qtype = (row.get(qtype_column) or "").strip() if qtype_column else ""
                batch.extend(
                    make_medquad_documents(question, answer, qtype, row_number)
                )
                if len(batch) >= EMBEDDING_BATCH_SIZE:
                    yield batch[:EMBEDDING_BATCH_SIZE]
                    batch = batch[EMBEDDING_BATCH_SIZE:]
    except Exception as error:
        raise ValueError(
            f"Could not read {csv_path.name}. Check that the file exists and is "
            f"a valid CSV. Details: {error}"
        ) from error

    if batch:
        yield batch


def create_embeddings() -> HuggingFaceEmbeddings:
    """Load the free embedding model on this computer (not through Gemini)."""
    print("Loading the local embedding model...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def connect_to_pinecone() -> tuple[Pinecone, object]:
    """Create the 384-dimensional index if needed, then connect to it."""
    api_key = require_environment_variable("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot-local").strip()
    if not index_name:
        index_name = "medical-chatbot-local"

    client = Pinecone(api_key=api_key)
    if index_name not in client.list_indexes().names():
        print(f"Creating Pinecone index '{index_name}'...")
        client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not client.describe_index(index_name).status["ready"]:
            print("Waiting for the Pinecone index to become ready...")
            time.sleep(2)
    else:
        description = client.describe_index(index_name)
        if description.dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"The existing index has dimension {description.dimension}, but "
                f"the local model needs {EMBEDDING_DIMENSION}. Set "
                "PINECONE_INDEX_NAME=medical-chatbot-local in .env."
            )
        if str(description.metric).lower() != "cosine":
            raise ValueError("The existing Pinecone index must use cosine similarity.")

    return client, client.Index(index_name)


def make_pdf_vector_id(document: Document, chunk_number: int) -> str:
    """Preserve the existing stable PDF ID strategy."""
    identity = (
        f"{document.metadata['filename']}|{document.metadata['page']}|"
        f"{chunk_number}|{document.page_content}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def upload_batch(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    index: object,
    uploaded_count: int,
    expected_count: int,
) -> int:
    """Embed and upload one manageable batch."""
    vectors = embeddings.embed_documents([doc.page_content for doc in documents])
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Local embedding dimension is {len(vector)} but Pinecone expects "
                f"{EMBEDDING_DIMENSION}."
            )

    records = []
    for offset, (document, vector) in enumerate(zip(documents, vectors)):
        vector_id = document.metadata.get("vector_id")
        if not vector_id:
            vector_id = make_pdf_vector_id(document, uploaded_count + offset)
        metadata = {
            key: value
            for key, value in document.metadata.items()
            if key != "vector_id" and value not in (None, "")
        }
        # MedQuAD already stores question and answer separately for clear retrieval.
        # PDF chunks store their text in one field.
        if document.metadata.get("document_type") == "pdf":
            metadata["text"] = document.page_content
        records.append({"id": vector_id, "values": vector, "metadata": metadata})

    # Embeddings are generated efficiently above, but Pinecone receives smaller
    # requests so each upsert stays safely below its request-size limit.
    for start in range(0, len(records), PINECONE_BATCH_SIZE):
        pinecone_batch = records[start : start + PINECONE_BATCH_SIZE]
        try:
            index.upsert(vectors=pinecone_batch)
        except Exception as error:
            error_text = str(error).lower()
            if "request size" in error_text or "2mb" in error_text:
                print(
                    "Pinecone batch was too large. Reduce PINECONE_BATCH_SIZE."
                )
            raise
        uploaded_count += len(pinecone_batch)
        print(f"Uploaded {uploaded_count:,} / at least {expected_count:,}")
    return uploaded_count


def main() -> None:
    """Index PDF chunks and MedQuAD rows without calling Gemini."""
    load_dotenv(PROJECT_DIR / ".env")
    try:
        pdf_chunks = split_pdf_documents(load_pdf_documents())
        csv_files = sorted(DATA_DIR.glob("*.csv"))
        csv_details = []
        csv_row_count = 0
        for csv_path in csv_files:
            columns, row_count = inspect_csv(csv_path)
            csv_details.append((csv_path, columns))
            csv_row_count += row_count

        if not pdf_chunks and not csv_details:
            raise ValueError("No PDF or CSV files were found inside the data folder.")

        expected_count = len(pdf_chunks) + csv_row_count
        embeddings = create_embeddings()
        _, index = connect_to_pinecone()
        print("Creating local embeddings (Gemini is not used during ingestion)...")

        uploaded_count = 0
        for start in range(0, len(pdf_chunks), EMBEDDING_BATCH_SIZE):
            uploaded_count = upload_batch(
                pdf_chunks[start : start + EMBEDDING_BATCH_SIZE],
                embeddings,
                index,
                uploaded_count,
                expected_count,
            )

        for csv_path, columns in csv_details:
            for batch in iter_csv_batches(csv_path, columns):
                uploaded_count = upload_batch(
                    batch, embeddings, index, uploaded_count, expected_count
                )

        print(f"Indexing completed successfully. Uploaded {uploaded_count:,} records.")
    except Exception as error:
        print(f"\nError: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
