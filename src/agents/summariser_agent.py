# src/agents/summarizer_agent.py

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
    final_answer: Optional[str] = None


def summarizer_agent(state: StateSchema) -> StateSchema:
    """
    Summarizes the validated or retrieved documents into a final answer.
    """
    question = state.question
    documents = state.validated_chunks or state.retrieved_docs

    if not question:
        raise ValueError("Missing 'question' in state")
    if not documents:
        raise ValueError("Missing documents to summarize")

    # Combine content
    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_template(
        "You are a financial analyst.\n\n"
        "Context:\n{context}\n\n"
        "Given the above context, answer the following question:\n"
        "{question}"
    )

    llm = ChatOpenAI(temperature=0)
    chain: Runnable = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    answer = response.content.strip() if hasattr(response, "content") else str(response).strip()

    # Return new state with final_answer added
    base_state = state.dict(exclude={"final_answer"})
    return StateSchema(**base_state, final_answer=answer)
