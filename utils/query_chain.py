from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def initialize_qa_chain(vector_store, openai_api_key):
    """Initializes a RetrievalQA chain."""
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain