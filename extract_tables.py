import pdfplumber
import pandas as pd
import re
import sqlite3
import os


def clean_column_name(col):

    col = col.replace("\n", " ")

    col = col.strip()

    # remove non-ascii characters
    col = col.encode("ascii", "ignore").decode()

    # remove special characters
    col = re.sub(r"[^\w\s]", "", col)

    # replace spaces with underscores
    col = col.replace(" ", "_")
    col = col.strip("_")

    return col


def clean_cell(x):

    if isinstance(x, str):

        x = x.replace("\n", " ")

        x = " ".join(x.split())

        # remove non-ascii characters
        x = x.encode("ascii", "ignore").decode()

    return x


def extract_pdf_tables(pdf_path):

    tables = []

    pdf_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    db_path = f"{pdf_name}.db"

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf.pages):

            extracted_tables = page.extract_tables()

            for table_index, table in enumerate(extracted_tables):

                if not table:
                    continue

                df = pd.DataFrame(
                    table[1:],
                    columns=table[0]
                )

                # clean column names
                df.columns = [
                    clean_column_name(col)
                    for col in df.columns
                ]

                # clean cell values
                df = df.map(clean_cell)

                # auto-detect numeric columns: strip commas/currency,
                # try to_numeric, and keep the conversion if most values parse.
                for col in df.columns:

                    stripped = (
                        df[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace(r"[^\d\.\-eE]", "", regex=True)
                    )

                    converted = pd.to_numeric(stripped, errors="coerce")

                    non_empty = df[col].astype(str).str.strip().ne("").sum()

                    if non_empty == 0:
                        continue

                    parse_rate = converted.notna().sum() / non_empty

                    if parse_rate >= 0.8:
                        df[col] = converted

                tables.append({
                    "page": page_number,
                    "table_index": table_index,
                    "dataframe": df
                })

    conn = sqlite3.connect(db_path)

    table_metadata = []

    for table_info in tables:

        page = table_info["page"]

        table_index = table_info["table_index"]

        df = table_info["dataframe"]

        table_name = f"page_{page}_table_{table_index}"

        df.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False
        )

        table_metadata.append({
            "table_name": table_name,
            "page": page,
            "columns": list(df.columns)
        })

    conn.close()

    return {
        "table_metadata": table_metadata,
        "db_path": db_path
    }