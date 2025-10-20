from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("----GENERATION----")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke(
        {
            "question": question,
            "context": documents,
        }
    )
    return {"question": question, "documents": documents, "generation": generation}
