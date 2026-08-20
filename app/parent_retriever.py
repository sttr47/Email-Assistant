from langchain_classic.retrievers import ParentDocumentRetriever

def build_parent_retriever(vectorstore, docstore, parent_splitter, child_splitter):
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        id_key="doc_id",
        search_kwargs={"k": 5}
    )