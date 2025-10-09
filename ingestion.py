import os
import asyncio
import ssl
import certifi
from typing import List, Any, Dict
from pinecone import Pinecone # Core Pinecone client
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings # for embedding

# Assume logger import is correct
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)


# --- GLOBAL INITIALIZATION ---

DOTENV_PATH = "Langchain/documentation-helper/.env"
load_dotenv(DOTENV_PATH)

#configure SSL context to use certifi certification
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

# Instantiate clients globally
try:
    # Tavily Crawl Client
    crawler = TavilyCrawl(max_depth=1, extract_depth="advanced") #, max_bredth = 20, max_pages=100)

    # Pinecone/Embeddings Setup
    INDEX_NAME = os.environ['INDEX_NAME'] 
    
    # Mistral Embeddings (1024 dimension Pinecone index)
    #embeddings = MistralAIEmbeddings(model="mistral-embed") #, chunk_size=50)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
    
    # Pinecone Client initialization
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"]) 
    
    # VectorStore instance
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

except KeyError as e:
    log_error(f"FATAL ERROR: Missing environment variable {e}. Check .env file.")
    exit()

# --- ASYNC BATCH INDEXING FUNCTION ---

async def index_documents_async(documents: list[Document], batch_size: int = 50):
    """Process documents in batches asynchronously using aadd_documents."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"Vectorstore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )
    
    # Create batches
    batches = [
        documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    
    log_info(
        f"Vectorstore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Inner async function to handle one batch
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            # Use the globally defined vectorstore instance
            await vectorstore.aadd_documents(batch) 
            log_success(
                f"Vectorstore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            ) 
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    
        
    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    # Gather tasks and capture exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True) 

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"Vectorstore Indexing: All batches processed successfully ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


# --- MAIN ORCHESTRATION FUNCTION ---

async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/",
        Colors.PURPLE
    )
    
    # 1. Crawl the documentation site
    # This uses the globally defined 'crawler' instance
    try:
        # Pass only the required URL to the runnable instance
        res = await crawler.ainvoke({"url": "https://python.langchain.com/"}) 
        
        # Ensure 'results' key exists and process
        if not isinstance(res, dict) or 'results' not in res:
             raise ValueError("Invalid response format from crawler.")

    except Exception as e:
        log_error(f"Tavily Crawl Failed: {e}")
        return # Stop pipeline on crawl failure

    all_docs = [
        Document(
            page_content=result['raw_content'], 
            metadata={"source": result['url']}
        ) 
        for result in res['results']
    ]
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site"
    )
    
    # 2. Chunk Documents
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap", 
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )
    
    # 3. Process and Index Documents
    # Pass the splitted documents to the async indexing function
    await index_documents_async(splitted_docs, batch_size=500)

    log_header ("PIPELINE COMPLETE")
    log_success("Ingestion Pipeline finished successfully")
    log_info("Summary:", Colors.BOLD)
    log_info(f" . URLs Crawled: {len(res['results'])}", Colors.BOLD)
    log_info(f" . Documents Extracted: {len(all_docs)}", Colors.BOLD)
    log_info(f" . Chunks Created: {len(splitted_docs)}", Colors.BOLD)

    
if __name__=="__main__":
    asyncio.run(main())