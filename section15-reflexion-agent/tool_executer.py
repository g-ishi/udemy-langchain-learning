from typing import List
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool

from schema import AnswerQuestion, ReviseAnswer

load_dotenv()


tavily_tool = TavilySearch(max_results=2)


def run_queries(search_queries: List[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


# 区別のために同じツールに別名をつける
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
