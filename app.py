import streamlit as st
from utils.file_processor import process_documents
from utils.vector_store import build_vector_store
from utils.query_chain import initialize_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai  # Import openai
import tiktoken  # Import tiktoken for token counting

# Load API Key from Streamlit secrets
OPENAI_API_KEY = st.secrets["general"]["OPENAI_API_KEY"]

# Function to estimate token count
def count_tokens(text):
    encoding = tiktoken.get_encoding("gpt2")  # You can use "gpt2" for most models
    return len(encoding.encode(text))

# Initialize the app
st.set_page_config(page_title="Long RAG App", layout="wide")
st.title("Long RAG App")
st.write("Upload documents and query them using Retrieval-Augmented Generation (RAG).")

# Upload documents with more file types: PDF, JPG, PNG, CSV, Excel
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, JPG, PNG, CSV, Excel)",
    type=["pdf", "jpg", "jpeg", "png", "csv", "xls", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write("Processing documents...")
    documents = process_documents(uploaded_files)

    # Display content of each document (truncated)
    for i, doc in enumerate(documents):
        st.subheader(f"Document {i+1}")
        st.text_area(f"Content of Document {i+1}", doc[:1000] + "...", height=200)

    # Build vector store
    st.write("Building vector store...")
    vector_store = build_vector_store(documents, OPENAI_API_KEY)

    # Initialize QA Chain
    qa_chain = initialize_qa_chain(vector_store, OPENAI_API_KEY)

    # Query input
    query = st.text_input("Ask a question about the uploaded documents:")

    # Execute query
    if query:
        st.write("Querying the documents...")

        # Split documents into smaller chunks to avoid token limit issues
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunked_docs = []

        for doc in documents:
            # Split each document into chunks
            chunks = text_splitter.split_text(doc)
            # Only keep chunks that fit within the model's token limit
            for chunk in chunks:
                # Ensure chunk doesn't exceed the token limit
                if count_tokens(chunk) < 15000:  # Leave some space for the query
                    chunked_docs.append(chunk)
                else:
                    st.warning(f"A chunk from the document is too large and has been discarded.")

        # Use the input query directly in the prompt
        prompt = f"Answer the following question based on the provided documents: {query}"

        responses = []
        for chunk in chunked_docs:
            try:
                # Run the query on the document chunk
                response = qa_chain.run(f"{prompt}\n{chunk}")
                
                # Clean up the response to ensure clarity
                cleaned_response = response.strip()
                responses.append(cleaned_response)
            except Exception as e:  # Catching all exceptions
                st.error(f"Error processing chunk: {e}")

        # Combine all responses and display the final summary
        final_summary = " ".join(responses).strip()
        st.subheader("Response")
        
        if final_summary:
            st.write(final_summary)
        else:
            st.write("No relevant information found in the documents for the provided question.")

else:
    st.write("Please upload documents to get started.")
