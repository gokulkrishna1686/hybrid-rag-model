from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import BaseModel
import os
import shutil
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document

from extract_images import extract_and_caption_images
from extract_tables import extract_pdf_tables

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

file_name = "data_files/Employee Performance.pdf"

result = extract_pdf_tables(
    file_name
)
table_metadata = result["table_metadata"]
db_path = result["db_path"]

loader = PyPDFLoader(file_name)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    add_start_index=True
)

chunks = text_splitter.split_documents(docs)

image_descriptions = extract_and_caption_images(file_name)

image_chunks = [
    Document(
        page_content=item["description"],
        metadata={
            "source": item["image_path"],
            "type": "image_caption",
        },
    )
    for item in image_descriptions
]

chunks.extend(image_chunks)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

semantic_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        semantic_retriever
    ],
    weights=[0.4, 0.6]
)

db = SQLDatabase.from_uri(
    f"sqlite:///{db_path}",
    sample_rows_in_table_info=2
)

@tool
def get_table_metadata():
    """
    Returns available SQL tables: CREATE TABLE schema (with column types)
    plus a few sample rows per table so the agent can pick the right table
    and write correct SQL.
    """
    page_index = "\n".join(
        f"-- {m['table_name']} (page {m['page']}): columns = {m['columns']}"
        for m in table_metadata
    )

    schema = db.get_table_info()

    payload = f"PAGE INDEX:\n{page_index}\n\nSCHEMA & SAMPLES:\n{schema}"

    print(f"METADATA RETRIEVED!")

    return payload

@tool
def query_pdf_tables(query: str):
    """
    Query structured PDF table data using SQL.
    """

    print("QUERY:", query)

    try:
        result = db.run(query)
        return str(result)
    
    except Exception as e:
        return f"SQL ERROR: {str(e)}"

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""

    retrieved_docs = hybrid_retriever.invoke(query)

    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata}\n"
            f"Content: {doc.page_content}"
        )
        for doc in retrieved_docs
    )
    print("CONTEXT RETRIVED!")

    return serialized, retrieved_docs

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
    api_key=api_key,
    max_completion_tokens=1000
)

class ResponseFormat(BaseModel):
    message: str

tools = [
    retrieve_context,
    query_pdf_tables,
    get_table_metadata
]

# prompt = """
# You answer questions about uploaded company reports.

# Tool usage rules:

# 1. retrieve_context(query: str)
# - Use ONLY for semantic document retrieval, summaries, explanations, charts, images, and general report context.
# - DO NOT use this for exact table calculations or SQL-style queries.

# 2. get_table_metadata()
# - Returns SQL table schemas and columns.
# - ALWAYS use this before generating SQL queries.

# 3. query_pdf_tables(query: str)
# - Use for structured table data.
# - Use for calculations, averages, filtering, counts, rankings, IDs, salaries, budgets, and exact lookups.
# - Input must be a valid SQLite query.

# For table/numerical questions:
# 1. inspect schema
# 2. generate SQL
# 3. execute SQL

# Respond in under 30 words.
# """

prompt = """
You answer questions about uploaded company reports.

Tool usage rules:

1. get_table_metadata()
- Returns a page index, CREATE TABLE schemas (with column types), and sample rows per table.
- ALWAYS use this before generating SQL queries.
- Pick the table whose columns and sample rows match the question.

2. query_pdf_tables(query: str)
- Use for structured table data.
- Use for calculations, averages, filtering, counts, rankings, IDs, salaries, budgets, and exact lookups.
- Input must be a valid SQLite query.
- Trust the column types from the schema (numeric columns are REAL/INTEGER — no need to CAST or strip commas).

3. retrieve_context(query: str)
- Use ONLY for semantic document retrieval, summaries, explanations, charts, images, and general report context.
- DO NOT use this for exact table calculations or SQL-style queries.

For table/numerical questions:
1. inspect schema
2. generate SQL
3. execute SQL

Respond in under 30 words.
"""

agent = create_agent(model=model, tools=tools, response_format=ToolStrategy(ResponseFormat), system_prompt=prompt)

while True:

    user_query = input("\nYou: ")

    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Exiting...")
        break

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        }
    )

    print("\nAssistant:", result["structured_response"].message)