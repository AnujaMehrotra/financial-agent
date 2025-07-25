# src/graph_builder.py

from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import Dict, List, Any
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Importing agent functions from their respective modules
from src.agents.retriever_agent import retriever_agent
from src.agents.analyst_agent import analyst_agent
from src.agents.validator_agent import validator_agent
from src.agents.summariser_agent import summarizer_agent


# --- Define State Schema ---
class StateSchema(BaseModel):
    question: str
    retrieved_docs: List[Any] = []
    analysis: str = ""
    validation: str = ""
    final_answer: str = ""


# --- Build the LangGraph ---
def build_graph():
    graph = StateGraph(StateSchema)

    graph.add_node("retriever", retriever_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("validator", validator_agent)
    graph.add_node("summarizer", summarizer_agent)

    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "validator")
    graph.add_edge("validator", "summarizer")

    graph.set_entry_point("retriever")
    graph.set_finish_point("summarizer")

    return graph.compile()
