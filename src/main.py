# src/main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_builder import build_graph
from src.graph_builder import StateSchema

if __name__ == "__main__":
    graph = build_graph()

    input_state = StateSchema(question="What is Tesla's AI strategy?")
    result = graph.invoke(input_state)

    if "final_answer" in result:
        print("\nFinal Answer:\n", result["final_answer"])
    else:
        print("No final answer generated.")

