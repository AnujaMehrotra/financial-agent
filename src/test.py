from pydantic import BaseModel
from langgraph.graph import StateGraph

class StateSchema(BaseModel):
    question: str
    retrieved_docs: list = []
    analysis: str = None
    validation: bool = None
    final_answer: str = None

# Sample dummy functions representing agents
def retriever(state):
    # pretend to fetch documents
    state.retrieved_docs = ["Doc 1 about Tesla", "Doc 2 about AI strategy"]
    return state

def analyst(state):
    state.analysis = "Analyzed documents and extracted insights."
    return state

def validator(state):
    state.validation = True
    return state

def summarizer(state):
    state.final_answer = "Tesla's AI strategy is focused on autonomous driving."
    return state

def build_graph():
    graph = StateGraph(state_schema=StateSchema)

    # Add steps/nodes as per the new API
    graph.add_state_handler(retriever)
    graph.add_state_handler(analyst)
    graph.add_state_handler(validator)
    graph.add_state_handler(summarizer)

    return graph

if __name__ == "__main__":
    g = build_graph()
    print(dir(g))
    state = StateSchema(question="What is Tesla's AI strategy?")
    result = g.run(state)
    
    print(result.final_answer)
