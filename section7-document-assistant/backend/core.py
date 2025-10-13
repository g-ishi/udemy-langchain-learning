import os
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from openai import embeddings
from langchain_core.documents import Document
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RunLLMResult(TypedDict):
    query: str
    result: str
    source_documents: list[Document]


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> RunLLMResult:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
    )
    index = pc.Index(os.environ["SECTION_7_INDEX_NAME"])

    docsearch = PineconeVectorStore(index=index, embedding=embeddings)

    retrieval_qa_chat_prompt: PromptTemplate = hub.pull(
        "langchain-ai/retrieval-qa-chat",
    )

    stuff_documents_chain = create_stuff_documents_chain(
        llm=ChatOpenAI(model="gpt-4o"),
        prompt=retrieval_qa_chat_prompt,
    )

    # 過去の会話履歴に関連する質問を独立した質問にリフレーズ
    rephrase_prompt: PromptTemplate = hub.pull(
        "langchain-ai/chat-langchain-rephrase",
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=ChatOpenAI(model="gpt-4o"),
        retriever=docsearch.as_retriever(),
        prompt=rephrase_prompt,
    )

    # qa = create_retrieval_chain(
    #     retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    # )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke({"input": query, "chat_history": chat_history})
    new_result: RunLLMResult = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    # query = "What is LangChain?"
    query = "How to make a pizza?"
    result = run_llm(query)
    print(result)
