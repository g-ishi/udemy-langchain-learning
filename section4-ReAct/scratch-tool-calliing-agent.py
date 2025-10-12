from operator import call
from typing import Union, List
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.schema import AgentAction, AgentFinish
from langchain_core.tools import Tool, BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.schema.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage


from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the input text."""
    print(f"Calculating length of text: {text}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: list[BaseTool], name: str) -> BaseTool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool with name {name} not found.")


if __name__ == "__main__":
    tools = [get_text_length]
    # print(prompt.format(input="What is the length of the text 'Hello, world!'?"))

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        callbacks=[AgentCallbackHandler()],
    )
    llm_with_tools = llm.bind_tools(tools)

    messages: List[BaseMessage] = [
        HumanMessage(content="What is the length of the text: DOG ?")
    ]

    while True:
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_input = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_input)
                print(f"Observation: {observation}")

                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_call_id,
                    )
                )
            continue

        # No tool calls, we are done
        print(f"Final Answer: {ai_message.content}")
        break
