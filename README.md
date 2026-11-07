# DataPulse AI — Enterprise Data Intelligence Copilot

A local-first GenAI portfolio project combining data engineering, data analytics, natural-language-to-SQL, and retrieval-augmented generation (RAG).

## Features

- CSV ingestion pipeline with basic cleaning and PostgreSQL loading.
- Business overview dashboard with metrics and Plotly charts.
- Natural-language questions translated into read-only PostgreSQL queries.
- SQL safety checks (single SELECT/WITH statement; blocked write/DDL keywords).
- Document ingestion, chunking, local Ollama embeddings, cosine-similarity retrieval, and grounded answers.
- Docker Compose deployment: Streamlit + PostgreSQL + Ollama.

## Architecture

`CSV / TXT -> Streamlit -> ETL / RAG -> PostgreSQL`
`Natural language -> Ollama LLM -> SQL validation -> PostgreSQL -> Plotly`

## Prerequisites

- Windows 10/11 with Docker Desktop installed and running.
- Docker Desktop configured to use Linux containers / WSL 2 backend.
- At least 8 GB RAM recommended; 12 GB or more is more comfortable for local models.
- Git and VS Code recommended.

## Run with Docker

Open PowerShell in this project folder:

```powershell
docker compose up --build -d
docker compose ps
```

Download the local models (first run only):

```powershell
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull nomic-embed-text
```

Open the app:

`http://localhost:8501`

The first model download can take several minutes and requires internet access. After the models are downloaded, inference runs locally.

## Load the sample data

In the app sidebar:
1. Upload `data/sample_sales.csv`.
2. Set destination table to `sales`.
3. Click **Run ETL pipeline**.

## Index the sample policy document

In the sidebar:
1. Upload `documents/sales_policy.txt`.
2. Click **Index document**.
3. Open **Document RAG** and ask: `What is the return window?`

## Example questions

- What is total revenue by region?
- Which product has the highest total sales?
- Show the monthly revenue trend.
- How many orders are there per category?
- What is the return window?
- How long do refunds take after inspection?

Revenue is not stored as a column in the sample file. Use `quantity * unit_price` in SQL for revenue calculations.

## Useful commands

```powershell
# View logs
docker compose logs -f app
docker compose logs -f ollama

# Stop containers (data volumes remain)
docker compose down

# Stop and delete local database/model data
docker compose down -v

# Rebuild app after changing Python files
docker compose up --build -d
```

## Troubleshooting

**Docker daemon / engine not available**
- Start Docker Desktop and wait until it reports that the engine is running.
- In Docker Desktop, enable the WSL 2 engine if using the WSL backend.
- Retry `docker compose ps`.

**App cannot reach Ollama**
- Check `docker compose ps` and `docker compose logs ollama`.
- Pull both models using the commands above.
- The Ollama status indicator in the sidebar should show Connected.

**Dependency build fails**
- Run `docker compose build --no-cache app`.
- Check `docker compose logs app`.
- Confirm Docker Desktop has enough memory and internet access for pip.

**Port already in use**
- Change the left side of `8501:8501` in `compose.yaml`, for example `8502:8501`, then open `http://localhost:8502`.

## Important security note

This is an educational portfolio application. The SQL validator is a conservative demo guard, not a complete SQL security boundary. For production, use a dedicated read-only database user, SQL parser/allow-list, query timeouts, authentication, authorization, audit logging, and resource limits. Do not expose the app or database publicly as configured.

## Suggested one-month build plan

| Week | Deliverable |
|---|---|
| 1 | Docker setup, PostgreSQL schema, CSV ETL, data-quality checks |
| 2 | Streamlit dashboard, KPIs, SQL analysis, Plotly charts |
| 3 | Ollama integration, natural-language-to-SQL, document RAG |
| 4 | Testing, error handling, README, screenshots, GitHub, resume, demo video |

## Resume project description

**DataPulse AI — Enterprise Data Intelligence Copilot**  
Built a Dockerized GenAI analytics application using Python, Streamlit, PostgreSQL, Pandas, and Ollama. Developed a CSV-to-PostgreSQL ETL pipeline, interactive KPI dashboards, natural-language-to-SQL querying with read-only validation, and a RAG workflow for document question answering using local embeddings and similarity search.

Only include metrics (such as rows processed, query latency, or retrieval accuracy) after measuring them on your own tests.
