from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        print("*" * 20)
        print("LLM start:", prompts[0])

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print("*" * 20)
        print("LLM end:", response.generations[0][0].text)
