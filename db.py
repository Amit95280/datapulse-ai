import os
import re
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import URL

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://datapulse:datapulse_local_password@db:5432/datapulse"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                source VARCHAR(255) NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

def safe_identifier(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    if not name or not name[0].isalpha():
        name = "data_" + name
    return name[:60]

def load_csv(df: pd.DataFrame, table_name: str):
    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")
    df = df.copy()
    df.columns = [safe_identifier(str(c)) for c in df.columns]
    table = safe_identifier(table_name)
    df = df.drop_duplicates()
    missing_before = int(df.isna().sum().sum())
    # Normalize pandas/numpy values for SQLAlchemy.
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=1000, method="multi")
    return {
        "table": table,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicates_removed": "Applied drop_duplicates",
        "missing_cells": missing_before
    }

def list_tables():
    return sorted([t for t in inspect(engine).get_table_names()
                   if t != "document_chunks" and not t.startswith("pg_")])

def table_columns(table):
    if table not in list_tables():
        return []
    return [c["name"] for c in inspect(engine).get_columns(table)]

def run_readonly_sql(query: str):
    q = query.strip().rstrip(";")
    if not q:
        raise ValueError("SQL query is empty.")
    # Conservative guard: one read-only statement only.
    if ";" in q:
        raise ValueError("Multiple SQL statements are not allowed.")
    if not (q.lower().startswith("select ") or q.lower().startswith("with ")):
        raise ValueError("Only SELECT or WITH queries are allowed.")
    blocked = r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|call|execute)\b"
    if re.search(blocked, q, flags=re.IGNORECASE):
        raise ValueError("The query contains a prohibited SQL operation.")
    with engine.connect() as conn:
        result = conn.execute(text(q))
        rows = result.fetchmany(500)
        return pd.DataFrame(rows, columns=result.keys())
