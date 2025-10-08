from operator import call
from typing import Union
from dotenv import load_dotenv
from langchain_core.tools import render_text_description
from langchain.prompts import PromptTemplate
from langchain.agents import tool
from langchain.schema import AgentAction, AgentFinish
from langchain_core.tools import Tool, BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad.log import format_log_to_str

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
    sample_text = "Hello, world!"
    # length = get_text_length(sample_text)
    # print(f"The length of the text '{sample_text}' is {length}.")

    tools = [get_text_length]

    template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools=tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    # print(prompt.format(input="What is the length of the text 'Hello, world!'?"))

    llm = ChatOpenAI(
        temperature=0,
        stop=["\nObservation", "Observation", "Observation:"],
        model="gpt-4",
        callbacks=[AgentCallbackHandler()],
    )
    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):

        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the text 'Hello, world!'?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        # print("-" * 50)
        # print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool = find_tool_by_name(tools, agent_step.tool)
            observation = tool.invoke(agent_step.tool_input)
            # print(f"Observation: {observation}")

            intermediate_steps.append((agent_step, str(observation)))

            # print("-" * 50)
            # print(format_log_to_str(intermediate_steps))

        # print("-" * 50)
        # print(agent_step)

    if isinstance(agent_step, AgentFinish):
        pass
        print(f"Final Answer: {agent_step.return_values}")
