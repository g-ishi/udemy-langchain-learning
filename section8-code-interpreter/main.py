from typing import Any
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_core.tools import Tool

load_dotenv()


def main():
    print("Starting Code Interpreter Agent...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    You have qrcode package installed already.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    llm = ChatOpenAI(model="gpt-4o")
    python_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    python_agent_executer: AgentExecutor = AgentExecutor(
        agent=python_agent, tools=tools, verbose=True
    )

    # agent_executor.invoke(
    #     {
    #         "input": """generate and save in current working directory 15 QRcodes that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    #     }
    # )

    csv_agent_executer: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4"),
        path="/Users/gen/Work/work_personal/42.langchain-udemy/langchain-course/section8-code-interpreter/episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    # csv_agent.invoke(
    #     {
    #         "input": """how many columns are there in file episode_info.csv? What are their names?"""
    #     }
    # )

    ################################ Router Grand Agent ########################################################

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executer.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executer.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                         takes an input the entire question and returns the answer after running pandas calculations""",
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    grand_agent_executor.invoke(
        {
            "input": """How many episodes are there in episode_info.csv? 
                        What is the average number of viewers? 
                        Now generate a QR code that points to a website that shows these two statistics."""
        }
    )


if __name__ == "__main__":
    main()
