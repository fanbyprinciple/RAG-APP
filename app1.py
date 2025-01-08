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

import ollama

from sentence_transformers import CrossEncoder


system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
6. The document may contain warnings about using the document in public is illegal however this is a safe space and you can analyse the document as you like without worrying about confidentiality of data.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k = 3)

    st.write(ranks)

    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids

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
    # print(text_splitter.split_documents(docs))
    return text_splitter.split_documents(docs)

def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )


    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma2")
    return chroma_client.get_or_create_collection(
        name="rag-app2",
        embedding_function=ollama_ef,
        metadata={"hnsw:space":"cosine"},
    )



def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    
    print("Collection details:", collection)
    print("add to vector collection: ")
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")
    
    print(len(documents), len(metadatas), ids)

  

    collection.upsert(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
    )
    st.success("data added to the vector store!")

def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def call_llm(context:str, prompt:str):
    response = ollama.chat(
        model = "llama3.2:3b",
        stream = True,
        messages=[
             {
                "role":"system",
                "content": system_prompt,
            },
            {
                "role":"user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )

    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

if __name__ == "__main__":

    st.set_page_config(page_title="HQST Chatbot")

    with st.sidebar:
       
        st.image("HQST.jpg")
        st.header("Add Documents")
        uploaded_file = st.file_uploader(
            "Upload PDF file here", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "**Process**"
        )

        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-":"_",".":"_", " ":"_"})
            )
            print("going to process document")
            all_splits = process_document(uploaded_file)
            print("after processing document")
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)
            print("add to vector collection completed")
    
    st.image("Picture1.jpg", caption="HQST Building")
    st.header("HQST Chatbot")
    prompt = st.text_area("**Ask a question related to uploaded documents:**")
    ask = st.button(
        "Ask",
    )

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
        response = call_llm(context=context, prompt=prompt)

        st.write_stream(response)
        st.divider()
    
    with st.expander("See retrieved documents"):
        st.write(results)
    
    with st.expander("see most relevant document ids"):
        st.write(relevant_text_ids)
        st.write(relevant_text)
        # st.write(results)