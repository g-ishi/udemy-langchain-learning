from dotenv import load_dotenv

from graph.state_graph import graph

load_dotenv()

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(graph.invoke({"question": "what is agent memory?"}))  # type: ignore
