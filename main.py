from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
import hashlib
import tempfile
import traceback
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

# ── App Setup ─────────────────────────────────────────────────

app = FastAPI(title="Text to SQL + BI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://instant-bi.vercel.app","http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Gemini Setup ──────────────────────────────────────────────

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found in environment.")

client = genai.Client(api_key=api_key)

MODELS = ["gemini-2.5-flash"]

# In-memory store
file_store: dict = {}

# ── Helpers ───────────────────────────────────────────────────

def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def get_sql_from_gemini(question: str, schema_string: str, model: str) -> str:

    prompt = f"""
You are an expert in converting English questions to SQL queries.

Database schema:
{schema_string}

Rules:
- Return ONLY the SQL query
- No explanation

Question: {question}
"""

    response = client.models.generate_content(model=model, contents=prompt)

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("sql"):
            text = text[3:]
    return text.strip()


def get_dashboard_from_gemini(question: str, headers: list, sample_data: list, sql_results: list, model: str) -> dict:

    prompt = f"""
You are an expert data analyst.

Dataset columns: {", ".join(headers)}

Sample data: {json.dumps(sample_data[:5])}

Query results: {json.dumps(sql_results[:50])}

User question: {question}

Return JSON:

{{
"title": "...",
"summary": "...",
"insights": [],
"charts": [
    {{
        "type": "bar | line | pie | area",
        "title": "...",
        "xAxisKey": "...",
        "yAxisKey": "...",
        "description": "..."
    }}
]
}}

Return ONLY JSON.
"""

    response = client.models.generate_content(model=model, contents=prompt)
    text = response.text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text.strip())


def run_sql_query(sql: str, db_path: str):

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(sql)

    rows = cur.fetchall()

    col_names = [desc[0] for desc in cur.description]

    conn.close()

    return rows, col_names


def try_models(fn, *args):

    last_err = None

    for model in MODELS:
        try:
            return fn(*args, model)
        except Exception as e:
            last_err = e

    raise last_err


# ── Routes ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    contents = await file.read()

    fhash = file_hash(contents)

    if fhash in file_store:

        info = file_store[fhash]

        return {
            "file_id": fhash,
            "filename": file.filename,
            "row_count": info["row_count"],
            "columns": info["columns"],
            "cached": True,
        }

    try:

        import io
        from csv_to_sqlite import process_file

        file_like = io.BytesIO(contents)

        file_like.filename = file.filename

        db_path = tempfile.mktemp(suffix=".db")

        result = process_file(file_like, db_path=db_path)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {e}\n{traceback.format_exc()}",
        )

    column_names = result["original_columns"]

    sanitized_col_names = [c["name"] for c in result["columns"]]

    conn = sqlite3.connect(result["db_path"])
    cur = conn.cursor()

    cur.execute(f"SELECT * FROM '{result['table_name']}' LIMIT 20")

    rows = cur.fetchall()

    db_col_names = [d[0] for d in cur.description]

    conn.close()

    sample_data = [dict(zip(db_col_names, r)) for r in rows]

    file_store[fhash] = {
        "db_path": result["db_path"],
        "schema_string": result["schema_string"],
        "columns": column_names,
        "sanitized_columns": sanitized_col_names,
        "row_count": result["row_count"],
        "sample_data": sample_data,
    }

    return {
        "file_id": fhash,
        "filename": file.filename,
        "row_count": result["row_count"],
        "columns": column_names,
        "cached": False,
    }


class QueryRequest(BaseModel):
    file_id: str
    question: str


@app.post("/query")
async def query_data(body: QueryRequest):

    if body.file_id not in file_store:

        raise HTTPException(
            status_code=404,
            detail="file_id not found. Upload file first.",
        )

    info = file_store[body.file_id]

    try:

        sql = try_models(
            get_sql_from_gemini,
            body.question,
            info["schema_string"],
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"SQL generation failed: {e}",
        )

    try:

        rows, col_names = run_sql_query(sql, info["db_path"])

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"SQL execution failed: {e}\nGenerated SQL: {sql}",
        )

    sql_results = [dict(zip(col_names, row)) for row in rows]

    try:

        dashboard = try_models(
            get_dashboard_from_gemini,
            body.question,
            info["sanitized_columns"],
            info["sample_data"],
            sql_results,
        )

    except Exception:

        dashboard = None

    return {
        "question": body.question,
        "sql": sql,
        "columns": col_names,
        "row_count": len(sql_results),
        "results": sql_results,
        "dashboard": dashboard,
    }