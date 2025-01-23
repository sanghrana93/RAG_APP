from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def build_vector_store(documents, openai_api_key):
    """Creates a vector store using FAISS and OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(documents, embeddings)
    return vector_store
