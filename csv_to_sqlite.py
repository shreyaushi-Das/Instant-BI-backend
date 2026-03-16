import sqlite3
import pandas as pd
import os
import re


def _sanitize_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "col"


def _infer_sqlite_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series.dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series.dtype):
        return "REAL"
    return "TEXT"


def process_file(file_source, db_path="database.db", table_name=None) -> dict:
    # detect extension
    fname = file_source if isinstance(file_source, str) else getattr(file_source, "filename", "data")
    ext = os.path.splitext(fname)[1].lower()

    try:
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_source)
        else:
            try:
                df = pd.read_csv(file_source, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_source, encoding="latin-1")
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    # Derive table name from filename
    if table_name is None:
        table_name = _sanitize_name(os.path.splitext(os.path.basename(fname))[0]) or "data"
    # Sanitize column names
    original_columns = list(df.columns)
    df.columns = [_sanitize_name(c) for c in df.columns]

    # Infer types
    columns = [{"name": col, "type": _infer_sqlite_type(df[col])} for col in df.columns]

    # Write to SQLite
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    # Schema string for Gemini prompt
    col_list = ", ".join(f"{c['name']} {c['type']}" for c in columns)
    schema_string = f"TABLE: {table_name} | COLUMNS: {col_list}"

    return {
        "db_path": db_path,
        "table_name": table_name,
        "columns": columns,
        "schema_string": schema_string,
        "row_count": len(df),
        "original_columns": original_columns,
    }


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python csv_to_sqlite.py yourfile.csv")
    else:
        result = process_file(sys.argv[1])
        print(json.dumps(result, indent=2))
