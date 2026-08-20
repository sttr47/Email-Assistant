import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def pdf_loader():
    all_documents = []
    for path in glob.glob('docs/**/*.pdf', recursive=True):
        loader = PyPDFLoader(path)
        documents = loader.load()
        all_documents.extend(documents)

    for path in glob.glob('docs/**/*.txt', recursive=True):
        loader = TextLoader(path, encoding='utf-8')
        documents = loader.load()
        all_documents.extend(documents)

    print(f'Loaded {len(all_documents)} documents')
    return all_documents