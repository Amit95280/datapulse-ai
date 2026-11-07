
import pandas as pd
import plotly.express as px
import streamlit as st

from io import BytesIO
from pypdf import PdfReader
from sqlalchemy import text

from db import (
    engine,
    init_db,
    load_csv,
    list_tables,
    table_columns,
    run_readonly_sql,
)

from genai import (
    ask_llm,
    answer_with_rag,
    ingest_document,
    retrieve_context,
    ollama_status,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DataPulse AI | Enterprise Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

init_db()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def extract_pdf_text(pdf_bytes):
    """Extract readable text from an uploaded PDF."""
    reader = PdfReader(BytesIO(pdf_bytes))

    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append(
                f"Page {page_number}:\n{page_text}"
            )

    return "\n\n".join(extracted_pages)


def format_number(value):
    """Format numbers for dashboard metrics."""
    return f"{value:,.0f}"


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("📊 DataPulse AI")

st.markdown(
    """
    **Enterprise Data Intelligence & Analytics Copilot**

    Transform raw business data into actionable insights using
    Data Engineering, Business Analytics, SQL and Generative AI.
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🚀 DataPulse AI")
    st.caption("Enterprise Analytics Platform")

    st.divider()

    # SYSTEM STATUS
    st.subheader("⚙️ System health")

    try:
        st.success("PostgreSQL database connected")
    except Exception:
        st.error("Database connection issue")

    try:
        if ollama_status():
            st.success("Local AI engine connected")
        else:
            st.warning("Ollama is not reachable")
    except Exception:
        st.warning("Unable to verify AI engine")

    st.divider()

    # ETL PIPELINE
    st.subheader("📥 Data ingestion")

    csv_file = st.file_uploader(
        "Upload business dataset",
        type=["csv"],
        help="Upload a CSV file to start the ETL pipeline.",
        key="csv_uploader",
    )

    table_name = st.text_input(
        "Destination table name",
        value="sales",
        help="Use a simple table name, such as sales or inventory.",
    )

    if st.button(
        "▶ Run ETL pipeline",
        use_container_width=True,
        disabled=csv_file is None,
    ):

        try:
            if not table_name.strip().replace("_", "").isalnum():
                st.error("Enter a valid table name.")
            else:
                with st.spinner("Processing and loading business data..."):

                    df = pd.read_csv(csv_file)

                    report = load_csv(df, table_name.strip())

                st.success("ETL pipeline completed successfully!")

                st.metric(
                    "Records loaded",
                    format_number(report["rows"]),
                )

                st.json(report)

        except Exception as error:
            st.error(f"ETL pipeline failed: {error}")

    st.divider()

    # DOCUMENT INGESTION
    st.subheader("📚 Knowledge ingestion")

    doc_file = st.file_uploader(
        "Upload company document",
        type=["txt", "md", "pdf"],
        help="Supported formats: PDF, TXT and Markdown.",
        key="document_uploader",
    )

    if st.button(
        "📌 Index document",
        use_container_width=True,
        disabled=doc_file is None,
    ):

        try:
            with st.spinner("Extracting and indexing document..."):

                file_name = doc_file.name.lower()

                if file_name.endswith(".pdf"):
                    document_text = extract_pdf_text(
                        doc_file.getvalue()
                    )

                elif file_name.endswith(".txt") or file_name.endswith(".md"):
                    document_text = doc_file.getvalue().decode(
                        "utf-8",
                        errors="ignore",
                    )

                else:
                    document_text = ""

                if not document_text.strip():
                    st.warning(
                        "No readable text found. "
                        "If this is a scanned PDF, OCR may be required."
                    )

                else:
                    count = ingest_document(
                        doc_file.name,
                        document_text,
                    )

                    st.success(
                        f"Document indexed successfully: {count} chunks."
                    )

        except Exception as error:
            st.error(f"Document indexing failed: {error}")

    st.divider()

    st.caption("Local development environment")
    st.caption("Python • Streamlit • PostgreSQL • Ollama")


# --------------------------------------------------
# MAIN NAVIGATION
# --------------------------------------------------

tab_dashboard, tab_ask, tab_docs, tab_sql = st.tabs(
    [
        "📈 Analytics Dashboard",
        "💬 AI Data Assistant",
        "📚 Document Intelligence",
        "🧪 SQL Workspace",
    ]
)


# ==================================================
# TAB 1: ANALYTICS DASHBOARD
# ==================================================

with tab_dashboard:

    st.subheader("📈 Business Intelligence Dashboard")

    st.caption(
        "Explore datasets, monitor data quality and generate "
        "interactive business visualizations."
    )

    try:
        tables = list_tables()

        if not tables:

            st.info(
                "No datasets found. Upload a CSV file using the "
                "sidebar to begin your analytics workflow."
            )

        else:

            selected_table = st.selectbox(
                "Select business dataset",
                tables,
            )

            # Load selected table
            df = pd.read_sql(
                text(f'SELECT * FROM "{selected_table}" LIMIT 10000'),
                engine,
            )

            if df.empty:
                st.warning("The selected table contains no records.")

            else:

                # KPI METRICS
                total_rows = len(df)
                total_columns = len(df.columns)
                missing_values = int(df.isna().sum().sum())
                duplicate_rows = int(df.duplicated().sum())

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total records",
                    format_number(total_rows),
                )

                col2.metric(
                    "Total columns",
                    total_columns,
                )

                col3.metric(
                    "Missing values",
                    format_number(missing_values),
                )

                col4.metric(
                    "Duplicate rows",
                    format_number(duplicate_rows),
                )

                st.divider()

                # DATA PREVIEW
                st.markdown("### Dataset preview")

                st.dataframe(
                    df.head(100),
                    use_container_width=True,
                    hide_index=True,
                )

                # DATA QUALITY
                with st.expander("🔍 Data quality report"):

                    quality_df = pd.DataFrame({
                        "Column": df.columns,
                        "Data type": [
                            str(dtype) for dtype in df.dtypes
                        ],
                        "Missing values": [
                            int(value) for value in df.isna().sum()
                        ],
                        "Unique values": [
                            int(df[column].nunique())
                            for column in df.columns
                        ],
                    })

                    st.dataframe(
                        quality_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.divider()

                # INTERACTIVE VISUALIZATION
                st.markdown("### Interactive analytics")

                numeric_columns = (
                    df.select_dtypes(include="number")
                    .columns.tolist()
                )

                category_columns = [
                    column
                    for column in df.columns
                    if column not in numeric_columns
                ]

                if numeric_columns:

                    x_options = category_columns or numeric_columns

                    left, right = st.columns(2)

                    with left:
                        x_column = st.selectbox(
                            "Category / Group",
                            x_options,
                            key="dashboard_x",
                        )

                    with right:
                        y_column = st.selectbox(
                            "Business metric",
                            numeric_columns,
                            key="dashboard_y",
                        )

                    chart_df = (
                        df.groupby(
                            x_column,
                            dropna=False,
                        )[y_column]
                        .sum()
                        .reset_index()
                    )

                    st.plotly_chart(
                        px.bar(
                            chart_df,
                            x=x_column,
                            y=y_column,
                            title=f"{y_column} by {x_column}",
                            template="plotly_white",
                        ),
                        use_container_width=True,
                    )

                    # DISTRIBUTION
                    st.markdown("### Metric distribution")

                    selected_metric = st.selectbox(
                        "Select metric for distribution",
                        numeric_columns,
                        key="distribution_metric",
                    )

                    st.plotly_chart(
                        px.histogram(
                            df,
                            x=selected_metric,
                            title=f"Distribution of {selected_metric}",
                            template="plotly_white",
                        ),
                        use_container_width=True,
                    )

                else:
                    st.info(
                        "No numeric columns are available for "
                        "numeric visualizations."
                    )

                # DATE TREND
                date_columns = [
                    column
                    for column in df.columns
                    if "date" in column.lower()
                ]

                if date_columns and numeric_columns:

                    st.markdown("### Time-series analysis")

                    date_column = date_columns[0]

                    metric_column = st.selectbox(
                        "Select trend metric",
                        numeric_columns,
                        key="trend_metric",
                    )

                    try:
                        trend_df = df.copy()

                        trend_df[date_column] = pd.to_datetime(
                            trend_df[date_column],
                            errors="coerce",
                        )

                        trend_df = trend_df.dropna(
                            subset=[date_column]
                        )

                        trend_df["Month"] = (
                            trend_df[date_column]
                            .dt.to_period("M")
                            .astype(str)
                        )

                        trend_df = (
                            trend_df.groupby("Month")[metric_column]
                            .sum()
                            .reset_index()
                        )

                        st.plotly_chart(
                            px.line(
                                trend_df,
                                x="Month",
                                y=metric_column,
                                markers=True,
                                title=f"Monthly {metric_column} trend",
                                template="plotly_white",
                            ),
                            use_container_width=True,
                        )

                    except Exception as error:
                        st.warning(
                            f"Unable to generate time-series chart: {error}"
                        )

    except Exception as error:
        st.error(f"Dashboard error: {error}")


# ==================================================
# TAB 2: AI DATA ASSISTANT
# ==================================================

with tab_ask:

    st.subheader("💬 AI-Powered Data Assistant")

    st.caption(
        "Ask questions in natural language and generate SQL-based "
        "business insights from your database."
    )

    question = st.text_area(
        "Ask a question about your business data",
        placeholder=(
            "Example: Which store has the highest inventory?\n"
            "Example: Show the top 5 products by revenue."
        ),
        height=110,
    )

    if st.button(
        "✨ Analyze with AI",
        type="primary",
        use_container_width=True,
        key="ask_data",
    ):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            try:

                tables = list_tables()

                schema = "\n".join(
                    [
                        f"{table}: {', '.join(table_columns(table))}"
                        for table in tables
                    ]
                )

                if not schema:
                    st.warning(
                        "No database tables found. Upload a CSV first."
                    )

                else:

                    with st.spinner("AI is generating a SQL query..."):

                        sql = ask_llm(question, schema)

                    st.markdown("### Generated SQL")

                    st.code(sql, language="sql")

                    with st.spinner("Executing read-only SQL query..."):

                        result = run_readonly_sql(sql)

                    st.markdown("### Query results")

                    st.dataframe(
                        result,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # RESULT VISUALIZATION
                    if not result.empty and len(result.columns) >= 2:

                        numeric_result_columns = (
                            result.select_dtypes(include="number")
                            .columns.tolist()
                        )

                        if numeric_result_columns:

                            x_axis = next(
                                (
                                    column
                                    for column in result.columns
                                    if column not in numeric_result_columns
                                ),
                                result.columns[0],
                            )

                            st.plotly_chart(
                                px.bar(
                                    result,
                                    x=x_axis,
                                    y=numeric_result_columns[0],
                                    title="AI-generated data visualization",
                                    template="plotly_white",
                                ),
                                use_container_width=True,
                            )

                    # BUSINESS EXPLANATION
                    with st.spinner("Generating business explanation..."):

                        explanation = answer_with_rag(
                            (
                                "Explain these SQL query results in "
                                "clear business language.\n"
                                f"Question: {question}\n"
                                "Results: "
                                f"{result.head(30).to_json(orient='records')}"
                            ),
                            use_rag=False,
                        )

                    st.markdown("### 🧠 AI business insights")

                    st.write(explanation)

            except Exception as error:
                st.error(f"AI analysis failed: {error}")


# ==================================================
# TAB 3: DOCUMENT INTELLIGENCE / RAG
# ==================================================

with tab_docs:

    st.subheader("📚 Enterprise Document Intelligence")

    st.caption(
        "Search company documents and retrieve context-aware "
        "answers using Retrieval-Augmented Generation (RAG)."
    )

    st.info(
        "Upload and index PDF, TXT or Markdown documents "
        "from the sidebar before asking questions."
    )

    doc_question = st.text_input(
        "Ask a question about company documents",
        placeholder="Example: What is the company's refund policy?",
    )

    if st.button(
        "🔍 Search knowledge base",
        type="primary",
        use_container_width=True,
    ):

        if not doc_question.strip():

            st.warning("Please enter a document question.")

        else:

            try:

                with st.spinner("Searching indexed documents..."):

                    context = retrieve_context(doc_question)

                if not context:

                    st.info(
                        "No relevant document content found. "
                        "Please index a document first."
                    )

                else:

                    with st.expander("📄 Retrieved document context"):

                        st.write(context)

                    with st.spinner(
                        "Generating an answer using retrieved context..."
                    ):

                        answer = answer_with_rag(
                            doc_question,
                            use_rag=True,
                            context=context,
                        )

                    st.markdown("### 🧠 AI-generated answer")

                    st.write(answer)

            except Exception as error:
                st.error(f"Document search failed: {error}")


# ==================================================
# TAB 4: SQL WORKSPACE
# ==================================================

with tab_sql:

    st.subheader("🧪 SQL Analytics Workspace")

    st.caption(
        "Run read-only SQL queries against your PostgreSQL database."
    )

    st.warning(
        "Only SELECT and WITH queries should be allowed. "
        "Do not use this workspace with an unrestricted database account."
    )

    sql_input = st.text_area(
        "Write your SQL query",
        value="SELECT * FROM sales LIMIT 10",
        height=150,
    )

    if st.button(
        "▶ Execute SQL",
        type="primary",
        use_container_width=True,
    ):

        if not sql_input.strip():

            st.warning("Enter a SQL query.")

        else:

            try:

                with st.spinner("Executing SQL query..."):

                    result = run_readonly_sql(sql_input)

                st.success(
                    f"Query executed successfully. "
                    f"{len(result):,} rows returned."
                )

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    label="⬇ Download results as CSV",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="datapulse_query_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as error:
                st.error(f"SQL execution failed: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DataPulse AI | Enterprise Data Intelligence & Analytics Copilot"
)

st.caption(
    "Built with Python, Streamlit, PostgreSQL, Pandas, Plotly, "
    "Docker and local Generative AI."
)

st.caption(
    "Development environment only. Production deployment requires "
    "authentication, authorization, secure database permissions, "
    "query validation and monitoring."
)
