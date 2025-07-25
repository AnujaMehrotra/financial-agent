import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.rag import load_vectorstore
from typing import Dict, Any
from langchain_core.documents import Document
from pydantic import BaseModel
from typing import List, Optional

"""
This agent will:
    Load the FAISS index
    Accept a user question
    Retrieve the top-k relevant chunks using similarity search
    Return the retrieved documents (with metadata if present)
"""
# Define your state schema (or import it if defined centrally)
class StateSchema(BaseModel):
    question: str
    retrieved_docs: Optional[List[Document]] = None
    # Add other fields if needed

def retriever_agent(state: StateSchema) -> StateSchema:
    """
    Given a state with a 'question', returns top-k relevant docs from FAISS vectorstore.
    """
    question = state.question

    # Load the vectorstore (e.g., FAISS)
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Retrieve documents relevant to the question
    retrieved_docs = retriever.invoke(question)

    # Return a new StateSchema with updated field
    return state.copy(update={"retrieved_docs": retrieved_docs})