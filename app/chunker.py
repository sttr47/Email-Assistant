from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)