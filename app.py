#streamlit run app.py
import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    # os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?","!", " ", ""]
    )
    return text_splitter.split_documents(docs)

def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag-app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space":"cosine"},
    )



def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")
    
    collection.upsert(
        documents=documents,
        metadatas =metadatas,
        ids=ids,
    )
    st.success("data added to the vector store!")

if __name__ == "__main__":
    with st.sidebar:
        st.set_page_config(page_title="RAG question answer")
        st.header("Rag Question Answer")
        uploaded_file = st.file_uploader(
            "** Upload PDF file for QnA**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "**Process**"
        )

        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-":"_",".":"_", " ":"_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)
    
      st.header("Rag Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
        "Ask",
    )