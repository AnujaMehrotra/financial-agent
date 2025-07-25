from typing import Dict, Any, List, Optional
from pydantic import BaseModel 
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

# Define your state schema (or import it if defined centrally)
class StateSchema(BaseModel):
    question: str
    retrieved_docs: List[Document]
    analyzed_docs: List[Document] = []  # Add this field

def analyst_agent(state: StateSchema) -> StateSchema:
    """
    Analyzes retrieved documents to extract key insights related to the question.
    """
    question = state.question
    docs: List[Document] = state.retrieved_docs

    if not question:
        raise ValueError("Missing 'question' in state")
    if not docs:
        raise ValueError("Missing 'retrieved_docs' in state")

    prompt = ChatPromptTemplate.from_template(
        "You are a financial analyst.\n\n"
        "Given the question:\n{question}\n\n"
        "Analyze the following document chunk for any financial or strategic insights relevant to the question.\n\n"
        "Chunk:\n{chunk}\n\n"
        "Respond with the extracted insight or 'irrelevant' if none."
    )

    llm = ChatOpenAI(temperature=0)
    chain: Runnable = prompt | llm

    analyzed_docs = []

    for doc in docs:
        response = chain.invoke({"question": question, "chunk": doc.page_content})
        content = response.content.strip() if hasattr(response, "content") else str(response).strip()

        if content.lower() != "irrelevant":
            # Create a new Document object with just the insight
            analyzed_docs.append(Document(page_content=content, metadata=doc.metadata))

    return StateSchema(
    question=state.question,
    retrieved_docs=state.retrieved_docs,
    analyzed_docs=analyzed_docs
)
