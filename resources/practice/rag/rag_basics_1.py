import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import pickle
import faiss

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
faiss_index_path = os.path.join(current_dir, "db", "faiss_index.idx")

# Check if the FAISS index file already exists
if not os.path.exists(faiss_index_path):
    print("FAISS index does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("\n--- Creating embeddings ---")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    print("\n--- Finished creating embeddings ---")

    # Create the FAISS vector store
    print("\n--- Creating FAISS vector store ---")
    db = FAISS.from_documents(docs, embeddings)
    
    db.save_local("faiss-index")
