from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from app.embedder import embeddings

client = QdrantClient(path="./qdrant_db")
COLLECTION_NAME = 'pdf_docs'

def ensure_collection_exists(recreate=False):
    try:
        collections_response = client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]
    except Exception:
        existing_collections = []

    if recreate and COLLECTION_NAME in existing_collections:
        client.delete_collection(collection_name=COLLECTION_NAME)
        existing_collections.remove(COLLECTION_NAME)

    if COLLECTION_NAME not in existing_collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            ),
            sparse_vectors_config={
                'langchain-sparse': SparseVectorParams()
            }
        )

def get_vector_store():
    ensure_collection_exists(recreate=False)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=FastEmbedSparse(model_name='Qdrant/bm25'),
        retrieval_mode=RetrievalMode.HYBRID
    )