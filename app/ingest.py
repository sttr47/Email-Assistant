import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.doc_loader import pdf_loader
from app.vector_db import get_vector_store, ensure_collection_exists
from app.parent_retriever import build_parent_retriever
from app.chunker import parent_splitter, child_splitter
from app.local_store import PickleDocStore

def clear_old_stores():
    if os.path.exists("./qdrant_db"):
        shutil.rmtree("./qdrant_db", ignore_errors=True)
        print("qdrant_db cleaned")
    if os.path.exists("./docstore.pkl"):
        os.remove("./docstore.pkl")
        print("docstore.pkl cleaned")

def run_ingestion():
    clear_old_stores()
    
    ensure_collection_exists(recreate=True)
    vectorstore = get_vector_store()
    
    docstore = PickleDocStore("./docstore.pkl")

    retriever = build_parent_retriever(vectorstore, docstore, parent_splitter, child_splitter)

    docs = pdf_loader()
    if not docs:
        print("No documents found")
        return

    print(f"{len(docs)} documents loading...")
    retriever.add_documents(docs, ids=None)
    print("Ingestion successful!")

if __name__ == "__main__":
    run_ingestion()