from app.vector_db import get_vector_store
from app.parent_retriever import build_parent_retriever
from app.chunker import parent_splitter, child_splitter
from app.local_store import PickleDocStore

def get_retriever():
    vectorstore = get_vector_store()
    docstore = PickleDocStore("./docstore.pkl")
    return build_parent_retriever(vectorstore, docstore, parent_splitter, child_splitter)