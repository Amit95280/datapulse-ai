
import os
import re
import json
from io import BytesIO

import requests
import numpy as np

from sqlalchemy import text
from db import engine
from pypdf import PdfReader


# ==================================================
# CONFIGURATION
# ==================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://ollama:11434"
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "qwen2.5:3b"
)

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "nomic-embed-text"
)


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text_value):
    """
    Remove NULL and invalid control characters
    before embedding or PostgreSQL storage.
    """

    if text_value is None:
        return ""

    text_value = str(text_value)

    # Remove NULL and other unwanted control characters.
    text_value = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
        "",
        text_value
    )

    return text_value.strip()


# ==================================================
# OLLAMA STATUS
# ==================================================

def ollama_status():

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=3
        )

        return response.ok

    except requests.RequestException:
        return False


# ==================================================
# LLM CHAT
# ==================================================

def chat(
    prompt,
    system="You are a careful enterprise data analyst. Never invent data."
):

    prompt = clean_text(prompt)
    system = clean_text(system)

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["message"]["content"].strip()


# ==================================================
# NATURAL LANGUAGE TO SQL
# ==================================================

def ask_llm(question, schema):

    question = clean_text(question)
    schema = clean_text(schema)

    prompt = f"""
Generate one PostgreSQL SELECT query to answer the user's question.

Use only tables and columns in this schema.

Do not use:
INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
multiple statements, or SQL comments.

Return SQL only, without markdown fences or explanations.

Always use a reasonable LIMIT of at most 500
unless the query is an aggregate.

DATABASE SCHEMA:
{schema}

QUESTION:
{question}
"""

    raw = chat(
        prompt,
        system="You generate safe, read-only PostgreSQL SQL."
    )

    raw = re.sub(
        r"^```(?:sql)?|```$",
        "",
        raw.strip(),
        flags=re.IGNORECASE
    ).strip()

    return raw


# ==================================================
# TEXT EMBEDDINGS
# ==================================================

def embed(text_value):

    text_value = clean_text(text_value)

    if not text_value:
        raise ValueError("Cannot generate an embedding for empty text.")

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text_value
        },
        timeout=120
    )

    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    if not embeddings:
        raise ValueError("Ollama returned an empty embedding.")

    return embeddings[0]


# ==================================================
# DOCUMENT CHUNKING
# ==================================================

def chunks(text_value, size=900, overlap=120):

    text_value = clean_text(text_value)

    # Normalize whitespace.
    text_value = re.sub(
        r"\s+",
        " ",
        text_value
    ).strip()

    if not text_value:
        return []

    if overlap >= size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    output = []
    start = 0

    while start < len(text_value):

        end = min(
            len(text_value),
            start + size
        )

        chunk_text = clean_text(
            text_value[start:end]
        )

        if chunk_text:
            output.append(chunk_text)

        if end == len(text_value):
            break

        start = end - overlap

    return output


# ==================================================
# DOCUMENT INGESTION / INDEXING
# ==================================================

def ingest_document(source, content):

    source = clean_text(source)
    content = clean_text(content)

    if not content:
        raise ValueError(
            "The uploaded document contains no readable text."
        )

    document_parts = chunks(content)

    if not document_parts:
        return 0

    rows = []

    # Generate embeddings for document chunks.
    for part in document_parts:

        part = clean_text(part)

        vector = embed(part)

        rows.append(
            (
                source,
                part,
                json.dumps(vector)
            )
        )

    # Store document chunks in PostgreSQL.
    with engine.begin() as conn:

        # Remove previous version of the same document.
        conn.execute(
            text("""
                DELETE FROM document_chunks
                WHERE source = :source
            """),
            {
                "source": source
            }
        )

        # Insert new document chunks.
        for source_name, part, vector in rows:

            conn.execute(
                text("""
                    INSERT INTO document_chunks
                    (
                        source,
                        chunk_text,
                        embedding
                    )
                    VALUES
                    (
                        :source,
                        :chunk_text,
                        CAST(:embedding AS JSONB)
                    )
                """),
                {
                    "source": source_name,
                    "chunk_text": part,
                    "embedding": vector
                }
            )

    return len(rows)


# ==================================================
# DOCUMENT RETRIEVAL
# ==================================================

def retrieve_context(question, top_k=4):

    question = clean_text(question)

    if not question:
        return ""

    query_vector = np.array(
        embed(question),
        dtype=float
    )

    with engine.connect() as conn:

        rows = conn.execute(
            text("""
                SELECT source, chunk_text, embedding
                FROM document_chunks
            """)
        ).fetchall()

    scored = []

    for row in rows:

        vector = np.array(
            row.embedding,
            dtype=float
        )

        denominator = (
            np.linalg.norm(query_vector)
            * np.linalg.norm(vector)
        )

        score = (
            float(
                np.dot(query_vector, vector) / denominator
            )
            if denominator
            else 0.0
        )

        scored.append(
            (
                score,
                row.source,
                row.chunk_text
            )
        )

    scored.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    return "\n\n".join(
        (
            f"[Source: {source} | "
            f"Similarity: {score:.3f}]\n{chunk}"
        )
        for score, source, chunk in scored[:top_k]
    )


# ==================================================
# RETRIEVAL AUGMENTED GENERATION
# ==================================================

def answer_with_rag(
    question,
    use_rag=True,
    context=None
):

    question = clean_text(question)

    if use_rag and context is None:
        context = retrieve_context(question)

    context = clean_text(
        context or "No document context supplied."
    )

    prompt = f"""
Answer the question using only the context below.

If the answer is not present, say that the
indexed documents do not contain enough information.

Cite the source filename shown in the context when possible.

CONTEXT:
{context}

QUESTION:
{question}
"""

    return chat(prompt)


# ==================================================
# PDF TEXT EXTRACTION
# ==================================================

def extract_pdf_text(pdf_bytes):
    """
    Extract text from uploaded PDF bytes.

    Accepts bytes returned by Streamlit's getvalue().
    """

    try:

        pdf_stream = BytesIO(pdf_bytes)

        reader = PdfReader(pdf_stream)

        extracted_pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text() or ""

            page_text = clean_text(page_text)

            if page_text:

                extracted_pages.append(
                    f"Page {page_number}:\n{page_text}"
                )

        extracted_text = "\n\n".join(
            extracted_pages
        )

        return clean_text(extracted_text)

    except Exception as error:

        raise ValueError(
            f"Unable to extract PDF text: {error}"
        ) from error
