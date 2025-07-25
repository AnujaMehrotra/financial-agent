# src/retrieval/rag.py

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from src.utils.loader import load_and_split_pdf
import os
# Load API key
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_documents(pdf_path: str) -> list[Document]:
    """
    Loads and splits documents from a PDF into chunks.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def build_vectorstore(docs: list[Document]) -> FAISS:
    """
    Builds a FAISS vectorstore from the provided documents using OpenAI embeddings.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, index_path: str):
    vectorstore.save_local(index_path)


def load_vectorstore(index_path: str = "index") -> FAISS:
    """
    Loads a FAISS vectorstore from local storage.
    """
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

