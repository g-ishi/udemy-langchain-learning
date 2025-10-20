from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state: GraphState):
    print("----DECIDE_TO_GENERATE---")

    if state["web_search"]:
        print("----DECISION: Need web search------")
        return WEBSEARCH
    else:
        print("----DECISION: No need web_search------")
        return GENERATE


builder = StateGraph(GraphState)
builder.add_node(RETRIEVE, retrieve)
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(GENERATE, generate)
builder.add_node(WEBSEARCH, web_search)

builder.set_entry_point(RETRIEVE)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)
builder.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
builder.add_edge(WEBSEARCH, GENERATE)
builder.add_edge(GENERATE, END)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
