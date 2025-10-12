import os

from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from langchain import hub

load_dotenv()


def format_docs(docs: List[Document]) -> str:
    return "\n".join(
        [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )


if __name__ == "__main__":
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o",
    )

    # Initialize the embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    # Initialize the vector store
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )

    query = "What is Pinecone in machine learning?"

    # retrieval_qa_chat_prompt = hub.pull(
    #     "langchain-ai/retrieval-qa-chat",
    # )
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrival_chain = create_retrieval_chain(
    #     retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    # )

    # result = retrival_chain.invoke(input={"input": query})
    # print(result)

    # scratch implementation

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always say "Thanks for asking!" at the end of your answer.
    
    {context}
    Question: {question}
    Answer in Markdown:"""
    custom_rag_prompt = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm
    )

    # chainの先頭が辞書の場合は、両方にqueryが渡される
    res = rag_chain.invoke(query)
    print(res)
