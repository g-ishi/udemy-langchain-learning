import asyncio
import os
from re import split
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from pinecone import Pinecone

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# configure SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=1,  # for small documents, set chunk_size to 1
    retry_min_seconds=10,
)

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"],
)
index = pc.Index(os.environ["SECTION_7_INDEX_NAME"])
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(
    max_depth=5,
    max_breadth=20,
    max_pages=1000,
)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    print("starting main")
    log_header("Document Ingestion")

    log_info("Starting web crawling...")

    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": "https://python.langchain.com/",
            "max_depth": 3,
            "extract_depth": "advanced",
            "instructions": "Documentation relevant to AI agents",
            "respect_robots_txt": True,
        }
    )
    if tavily_crawl_results.get("error"):
        log_error(f"Crawling failed: {tavily_crawl_results['error']}")
        return
    else:
        log_success(f"Crawling completed. Found {len(tavily_crawl_results)} URLs.")

    all_docs = []
    for tavily_crawl_result_item in tavily_crawl_results["results"]:
        log_info(f"Extracting content from {tavily_crawl_result_item['url']}...")
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    log_header("Document chunking phase")
    log_info(
        f"Starting document chunking for {len(all_docs)} documents...", Colors.DARKCYAN
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(all_docs)
    await index_documents_async(split_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Pages crawled: {len(tavily_crawl_results)}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(split_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
