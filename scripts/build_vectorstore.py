# scripts/build_vectorstore.py

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

# Load API key
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Load raw text file
text_path = os.path.join("data", "tesla_10k_2023.txt")
loader = TextLoader(text_path, encoding="utf-8")
raw_docs = loader.load()

# ✅ Chunk the text into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(raw_docs)

# ✅ Generate embeddings
embeddings = OpenAIEmbeddings()

# ✅ Create vectorstore
vectorstore = FAISS.from_documents(docs, embeddings)

# ✅ Save to disk (into 'index' folder)
index_path = os.path.join("index")
vectorstore.save_local(index_path)

print("✅ FAISS vectorstore saved to:", index_path)
