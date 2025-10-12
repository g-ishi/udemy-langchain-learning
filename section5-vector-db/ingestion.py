import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    # Load the documents
    print("Loading documents...")
    loader = TextLoader(
        "/Users/gen/Work/work_personal/42.langchain-udemy/langchain-course/section5-vector-db/medium-blog.txt"
    )
    documents = loader.load()

    # Split the documents into chunks
    print("Splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # # Create the embeddings
    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    # # Create the vector store
    vectorstore = PineconeVectorStore.from_documents(
        docs, embeddings, index_name=os.environ["INDEX_NAME"]
    )

    # # Print the number of documents in the vector store
    print("finished.")
