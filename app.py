"""A simple Streamlit medical RAG chatbot."""

import os
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone


PROJECT_DIR = Path(__file__).parent
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
CHAT_MODEL = "gemini-3.8-flash"
FALLBACK_CHAT_MODEL = "gemini-3.7-flash"
TOP_K = 5
GEMINI_RETRY_DELAYS = (0, 2, 4, 8)
TRANSIENT_GEMINI_STATUS_CODES = {429, 500, 503, 504}

SYSTEM_PROMPT = """You are a medical information assistant.

Your job is to provide clear educational information based only on the supplied
medical context.

Rules:
- Do not claim to be a doctor.
- Do not provide a definitive medical diagnosis.
- Do not use outside knowledge or invent information unsupported by the context.
- If the context is insufficient, say exactly: "I don't have enough information
  in the available medical knowledge base to answer that confidently."
- Explain medical terms in simple language.
- Encourage consultation with a qualified healthcare professional when appropriate.
- For potentially serious symptoms, advise the user to seek urgent medical care.
- Never tell a user to stop, start, or change prescription medication without
  speaking to a qualified healthcare professional.

Keep the response understandable for a normal patient.
"""

EMERGENCY_PHRASES = (
    "severe chest pain",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "unconscious",
    "stroke symptoms",
    "severe bleeding",
    "seizure",
    "suicidal thoughts",
    "suicidal",
    "self-harm",
    "self harm",
)

EMERGENCY_MESSAGE = (
    "This may be a medical emergency. Please contact your local emergency services "
    "now or go to the nearest emergency department. If possible, ask someone you "
    "trust to stay with you. This chatbot cannot diagnose or safely manage an emergency."
)


def check_emergency(question: str) -> bool:
    """Look for a short list of obvious emergency phrases."""
    normalized_question = question.lower()
    return any(phrase in normalized_question for phrase in EMERGENCY_PHRASES)


def require_environment_variable(name: str) -> str:
    """Return a setting or show the exact action the beginner needs to take."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is missing from .env.")
    return value


@st.cache_resource
def create_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the free local model so Streamlit reuses it."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_pinecone_index() -> object:
    """Connect to the existing Pinecone knowledge-base index."""
    api_key = require_environment_variable("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot-local").strip()
    if not index_name:
        index_name = "medical-chatbot-local"

    client = Pinecone(api_key=api_key)
    if index_name not in client.list_indexes().names():
        raise ValueError("Your knowledge base appears to be empty. Run python ingest.py first.")

    index = client.Index(index_name)
    stats = index.describe_index_stats()
    if stats.total_vector_count == 0:
        raise ValueError("Your knowledge base appears to be empty. Run python ingest.py first.")
    return index


def retrieve_context(question: str) -> tuple[str, list[str]]:
    """Find five relevant PDF or MedQuAD records using local embeddings."""
    embeddings = create_embeddings()
    question_vector = embeddings.embed_query(question)
    if len(question_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Local embedding dimension is {len(question_vector)} but Pinecone "
            f"expects {EMBEDDING_DIMENSION}."
        )
    index = get_pinecone_index()
    result = index.query(vector=question_vector, top_k=TOP_K, include_metadata=True)

    context_parts: list[str] = []
    sources: list[str] = []
    for match in result.matches:
        metadata = match.metadata or {}
        if metadata.get("document_type") == "medquad":
            question_text = metadata.get("question", "")
            answer_text = metadata.get("answer", "")
            source = metadata.get("source", "")
            if question_text and answer_text:
                source_label = f"MedQuAD — {source}" if source else "MedQuAD dataset"
                context_parts.append(
                    "Medical reference:\n\n"
                    f"Question:\n{question_text}\n\n"
                    f"Answer:\n{answer_text}\n\n"
                    f"Source:\n{source_label}"
                )
                if source_label not in sources:
                    sources.append(source_label)
        else:
            text = metadata.get("text", "")
            filename = metadata.get("filename", "Unknown source")
            page = metadata.get("page")
            if text:
                source_label = f"{filename} — Page {page}" if page else filename
                context_parts.append(f"Source: {source_label}\n{text}")
                if source_label not in sources:
                    sources.append(source_label)

    if not context_parts:
        raise ValueError("Your knowledge base appears to be empty. Run python ingest.py first.")
    return "\n\n---\n\n".join(context_parts), sources


def extract_text_from_ai_message(response: object) -> str:
    """Return only user-visible text, never Gemini signatures or metadata."""
    fallback = "Sorry, I couldn't generate a readable response. Please try again."
    content = getattr(response, "content", None)

    # Older/simple responses contain one normal string.
    if isinstance(content, str):
        return content if content.strip() else fallback

    if not isinstance(content, list):
        return fallback

    text_parts: list[str] = []
    for block in content:
        # Dictionary blocks may also contain signatures or reasoning metadata.
        # Read only the text field from an explicitly user-visible text block.
        if isinstance(block, dict):
            if block.get("type") != "text":
                continue
            text = block.get("text")
        else:
            # LangChain may expose content blocks as objects instead of dicts.
            # If the object declares a type, accept only the "text" type.
            block_type = getattr(block, "type", None)
            if block_type not in (None, "text"):
                continue
            text = getattr(block, "text", None)

        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())

    return "\n\n".join(text_parts) if text_parts else fallback


class GeminiFriendlyError(Exception):
    """A safe Gemini error message that may be shown to the user."""


def get_gemini_status_code(error: Exception) -> int | None:
    """Read an HTTP/gRPC status without depending on raw exception JSON."""
    possible_values = [getattr(error, "status_code", None), getattr(error, "code", None)]
    for value in possible_values:
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, int):
            return value
        value_text = str(value).upper()
        named_codes = {
            "RESOURCE_EXHAUSTED": 429,
            "INTERNAL": 500,
            "UNAVAILABLE": 503,
            "DEADLINE_EXCEEDED": 504,
        }
        for name, status_code in named_codes.items():
            if name in value_text:
                return status_code

    match = re.search(r"\b(429|500|503|504)\b", str(error))
    return int(match.group(1)) if match else None


def safe_non_transient_gemini_error(error: Exception) -> GeminiFriendlyError:
    """Replace non-retryable Gemini details with a safe, useful message."""
    details = str(error).lower()
    if "api key" in details or "unauth" in details:
        return GeminiFriendlyError(
            "The Gemini API key was rejected. Check GEMINI_API_KEY in your .env file."
        )
    if "permission" in details or "forbidden" in details or "403" in details:
        return GeminiFriendlyError(
            "The Gemini request was denied. Check the API key's permissions."
        )
    if "malformed" in details or "invalid argument" in details or "400" in details:
        return GeminiFriendlyError(
            "Gemini could not process the request. Please try a different question."
        )
    return GeminiFriendlyError(
        "Gemini could not generate an answer. Please try again later."
    )


def invoke_gemini_with_retries(
    model: ChatGoogleGenerativeAI,
    messages: list[object],
    gemini_api_key: str,
) -> object:
    """Retry only temporary Gemini failures, then optionally try one fallback."""
    last_status_code: int | None = None

    for attempt, delay in enumerate(GEMINI_RETRY_DELAYS):
        try:
            if attempt == 0:
                return model.invoke(messages)
            with st.spinner("Medical AI is temporarily busy. Retrying..."):
                time.sleep(delay)
                return model.invoke(messages)
        except Exception as error:
            last_status_code = get_gemini_status_code(error)
            if last_status_code not in TRANSIENT_GEMINI_STATUS_CODES:
                raise safe_non_transient_gemini_error(error) from error

    # Use the fallback exactly once, and only when the primary model stayed
    # unavailable with status 503 through all four attempts.
    if last_status_code == 503:
        fallback_model = ChatGoogleGenerativeAI(
            model=FALLBACK_CHAT_MODEL,
            google_api_key=gemini_api_key,
            temperature=0.2,
        )
        try:
            with st.spinner("Medical AI is temporarily busy. Retrying..."):
                return fallback_model.invoke(messages)
        except Exception as error:
            fallback_status = get_gemini_status_code(error)
            if fallback_status == 429:
                raise GeminiFriendlyError(
                    "The Gemini API usage limit has been reached. Please try again later."
                ) from error
            if fallback_status in TRANSIENT_GEMINI_STATUS_CODES:
                raise GeminiFriendlyError(
                    "The medical AI service is temporarily busy. Please try again in a moment."
                ) from error
            raise safe_non_transient_gemini_error(error) from error

    if last_status_code == 429:
        raise GeminiFriendlyError(
            "The Gemini API usage limit has been reached. Please try again later."
        )
    raise GeminiFriendlyError(
        "The medical AI service is temporarily busy. Please try again in a moment."
    )


def generate_answer(question: str, context: str) -> str:
    """Ask the language model to answer using only the retrieved PDF text."""
    gemini_api_key = require_environment_variable("GEMINI_API_KEY")
    model = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=gemini_api_key,
        temperature=0.2,
    )
    user_prompt = f"""Medical context:
{context}

User question: {question}

Answer using only the medical context above. If it is not enough, use the exact
insufficient-information sentence from your instructions."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    response = invoke_gemini_with_retries(
        model=model,
        messages=messages,
        gemini_api_key=gemini_api_key,
    )
    return extract_text_from_ai_message(response)


def friendly_error_message(error: Exception) -> str:
    """Turn common Gemini errors into short, helpful messages."""
    if isinstance(error, GeminiFriendlyError):
        return str(error)
    details = str(error)
    normalized_details = details.lower()
    quota_words = ("quota", "rate limit", "resource_exhausted", "429")
    if any(word in normalized_details for word in quota_words):
        return "The Gemini API usage limit has been reached. Please try again later."
    if "api key" in normalized_details or "unauth" in normalized_details:
        return "The Gemini API key was rejected. Check GEMINI_API_KEY in your .env file."
    return f"I could not answer that question. {details}"


def show_sources(sources: list[str]) -> None:
    """Display each retrieved filename/page only once."""
    if sources:
        st.markdown("**Sources:**")
        for source in sources:
            st.markdown(f"- {source}")


def main() -> None:
    """Draw the page and handle each chat question."""
    load_dotenv(PROJECT_DIR / ".env")
    st.set_page_config(page_title="Medical AI Assistant", page_icon="🩺")
    st.title("Medical AI Assistant")
    st.write(
        "Ask general health and medical education questions based on the available "
        "medical knowledge base."
    )
    st.warning(
        "This chatbot provides educational information only and does not replace "
        "professional medical advice."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Redraw earlier messages whenever Streamlit reruns the page.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message.get("sources", []))

    question = st.chat_input("Ask a medical education question")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if check_emergency(question):
            answer = EMERGENCY_MESSAGE
            sources: list[str] = []
            st.error(answer)
        else:
            try:
                with st.spinner("Searching the medical knowledge base..."):
                    context, sources = retrieve_context(question)
                    answer = generate_answer(question, context)
                st.markdown(answer)
                show_sources(sources)
            except Exception as error:
                # Keep the interface readable instead of showing a Python traceback.
                answer = friendly_error_message(error)
                sources = []
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
