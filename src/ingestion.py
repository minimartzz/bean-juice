"""
ingestion.py - Document Ingestion Pipeline
==========================================
Module converts raw JSON coffee dataset into
searchable ChromaDB vector store.

Components:
- 
"""
import json
import os
import shutil
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CHROMA_DIR  = os.environ["CHROMA_DIR"]
JSON_PATH   = os.environ["JSON_PATH"]
COLLECTION  = "coffee"
EMBED_MODEL = os.environ["EMBED_MODEL"]

def load_coffee_documents() -> List[Document]:
  """
  Converts each JSON entry into LangChain Documents.

  The page_content of each Document combines the different key values of each
  entry into a single description of the coffee bean. Structred fields are also
  stored in metadata for downstream filtering.

  Returns:
    List[Document]: List of Langchain Documents
  """
  with open(JSON_PATH, "r") as f:
    coffees = json.load(f)
  
  documents = []
  for coffee in coffees:
    page_content = f"""
    The {coffee['bean']} is a bean from {coffee['Roaster Location']} which originated in {coffee['Coffee Origin']}.
    Roasted by {coffee['roaster']}. {coffee['Bottom Line']}

    OVERALL RATING: {coffee['rating']}

    Flavor Profile:
    Roast Level: {coffee['Roast Level']}
    Agtron: {coffee['agtron']}
    Aroma: {coffee['Aroma']}
    Body: {coffee['Body']}
    Flavor: {coffee['Flavor']}
    Aftertaste: {coffee['Aftertaste']}
    With Milk: {coffee['With Milk']}
    Acidity/ Structure: {coffee["Acidity/Structure"]}
    Acidity: {coffee["Acidity"]}

    Blind Assessment
    {coffee['Blind Assessment']}

    Notes
    {coffee['Notes']}

    Who Should Drink It
    {coffee['Who Should Drink It']}

    Coffee Details
    Price: {coffee['Est Price']}
    Review Date: {coffee['Review Date']}
    """.strip()

    doc = Document(
      page_content=page_content,
      metadata={
        "name": {coffee['bean']},
        "location": {coffee['Roaster Location']},
        "origin": {coffee['Coffee Origin']},
        "roaster": {coffee['roaster']},
        "roast_level": {coffee['Roast Level']},
        "agtron": {coffee['agtron']},
        "aroma": {coffee['Aroma']},
        "body": {coffee['Body']},
        "flavor": {coffee['Flavor']},
        "aftertaste": {coffee['Aftertaste']},
        "with Milk": {coffee['With Milk']},
        "acidity/ Structure": {coffee["Acidity/Structure"]},
        "acidity": {coffee["Acidity"]},
        "price": {coffee['Est Price']},
        "review_date": {coffee['Review Date']}
      }
    )
    documents.append(doc)
  
  print(f"[INGEST] loaded {len(documents)} coffee documents")
  return documents

def build_vectorstore(reset: bool = False) -> Chroma:
  """
  Load -> Chunk -> Embed -> Store. Process of converting documents
  into embeddings then stored in ChromaDB

  Args:
      reset (bool, optional): Rebuilds the ChromaDB from new coffee JSON. Defaults to False.

  Returns:
      Chroma: ChromaDB with coffee entries
  """
  if reset and os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
    print(f"[INGEST] Reset: Clearing existing existing ChromaDB at {CHROMA_DIR}")
  
  # ---- TextSplitter ------------------------------
  # chunk_size=1000 is safe for all-MiniLM-L6-v2 model
  # chunk_overlap=100 ensures sentences are kept on both ends
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True
  )

  docs = load_coffee_documents()
  chunks = splitter.split_documents(docs)
  print(f"[INGEST] Split into {len(chunks)} chunks")

  # ---- Embeddings ------------------------------
  print(f"[INGEST] Loading embedding model '{EMBED_MODEL}'...")
  embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
  )

  # ---- VectorStore ------------------------------
  print("[INGEST] Embedding and storing in ChromaDB...")
  vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION
  )
  count = vectorstore._collection.count()
  print(f"[INGEST] Stored {count} vectors at '{CHROMA_DIR}'")
  return vectorstore

def load_vectorstore() -> Chroma:
  if not os.path.exists(CHROMA_DIR):
    raise FileNotFoundError(
      "No ChromaDB found. Run: python ingestion.py"
    )
  embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
  )
  vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION
  )
  count = vectorstore._collection.count()
  print(f"[INGEST] Loaded {count} vectors from '{CHROMA_DIR}'")
  return vectorstore

if __name__ == "__main__":
  build_vectorstore(reset=True)