from json import load
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


load_dotenv()

if __name__ == "__main__":
    # Load the PDF
    loader = PyPDFLoader(
        "/Users/gen/Work/work_personal/42.langchain-udemy/langchain-course/section5-vector-db/ReAct_paper.pdf"
    )
    # すでにチャンク化されているが、チャンクサイズを変えたいので再度チャンク化する
    documents = loader.load()

    # Split the text into chunks
    # 再度チャンク化する
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents)

    # Create the embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    # Create the vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    query = "Give me a summary of the ReAct paper."

    retrieval_qa_chat_prompt = hub.pull(
        "langchain-ai/retrieval-qa-chat",
    )
    combine_docs_chain = create_stuff_documents_chain(
        ChatOpenAI(model="gpt-4o"), retrieval_qa_chat_prompt
    )
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke({"input": query})
    print(result)
