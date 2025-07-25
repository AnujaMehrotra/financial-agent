from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class StateSchema(BaseModel):
    question: str
    retrieved_docs: Optional[List[Document]] = None
    analyzed_docs: Optional[List[Document]] = None
    validated_chunks: Optional[List[Document]] = None


def validator_agent(state: StateSchema) -> StateSchema:
    """
    Validates retrieved or analyzed documents to ensure relevance/accuracy.

    Args:
        state (StateSchema): Must contain 'question' and either 'retrieved_docs' or 'analyzed_docs'.

    Returns:
        Updated StateSchema with validated_chunks added.
    """
    question = state.question
    docs: List[Document] = state.analyzed_docs or state.retrieved_docs

    if not docs:
        raise ValueError("Missing documents to validate")

    prompt = ChatPromptTemplate.from_template(
        "You are a strict compliance analyst.\n\n"
        "Question:\n{question}\n\n"
        "Document Chunk:\n{chunk}\n\n"
        "Is this chunk relevant and factually aligned with the question? Reply with 'yes' or 'no'."
    )
    llm = ChatOpenAI(temperature=0)
    chain: Runnable = prompt | llm

    validated_docs = []

    for doc in docs:
        response = chain.invoke({"question": question, "chunk": doc.page_content})
        is_valid = (
            response.content.strip().lower() == "yes"
            if hasattr(response, "content")
            else str(response).strip().lower() == "yes"
        )
        if is_valid:
            validated_docs.append(doc)

    # Remove the existing validated_chunks if any, to avoid duplicate kwargs
    base_state = state.dict(exclude={"validated_chunks"})
    return StateSchema(**base_state, validated_chunks=validated_docs)
