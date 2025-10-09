from dotenv import load_dotenv
import os

#DOTENV_PATH = "Langchain/documentation-helper/.env"
DOTENV_PATH = "/content/drive/My Drive/Colab Notebooks/Langchain/documentation-helper/.env"
load_dotenv(DOTENV_PATH)


from langchain.chains.retrieval import create_retrieval_chain 
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq 
from pinecone import Pinecone


def run_llm(query: str):
   
  #INDEX_NAME = os.environ['INDEX_NAME']
  #groq_key = os.environ["GROQ_API_KEY"]
  #pc = os.environ["PINECONE_API_KEY"]

  INDEX_NAME = os.getenv('INDEX_NAME')
  PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
  groq_key = os.getenv('GROQ_API_KEY')
  #PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  # Pinecone Client initialization
  pc = Pinecone(api_key=PINECONE_API_KEY)
  
      
  # VectorStore instance
  docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
  #chat = llm = ChatGroq(temperature=0, model="gemma2-9b-it", verbose = 0)
  chat = llm = ChatGroq(temperature=0, model="openai/gpt-oss-20b", verbose = 0)

  retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
  stuff_document_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

  qa = create_retrieval_chain(
    retriever = docsearch.as_retriever(), combine_docs_chain = stuff_document_chain
  )

  result = qa.invoke(input={"input": query})
  new_result = {
    "query": result["input"],
    "result": result["answer"],
    "source_documents": result["context"],
  }
  return new_result
    

    
if __name__=="__main__":
    res = run_llm(query = "What is a LangChain Chain?")
    print(res["result"])
