# src/utils/loader.py

from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

def build_vectorstore(documents: list[Document], save_path: str = "index"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"Vectorstore saved to {save_path}")

def load_and_split_pdf(path: str) -> List[Document]:
    """
    Loads a PDF file and splits it into manageable chunks using RecursiveCharacterTextSplitter.
    
    Args:
        path (str): Path to the PDF file.

    Returns:
        List[Document]: List of document chunks with metadata.
    """
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)
