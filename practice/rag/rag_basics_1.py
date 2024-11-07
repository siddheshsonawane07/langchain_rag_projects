import os
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "faiss_db_for_basics")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    os.makedirs(persistent_directory, exist_ok=True)
    print("Persistent directory does not exist. Initializing vector store...")

    specific_book = "odyssey.txt" 

    file_path = os.path.join(books_dir, specific_book)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {specific_book} does not exist in the directory {books_dir}."
        )

    loader = TextLoader(file_path, encoding='utf-8')
    book_docs = loader.load()

    documents = []
    for doc in book_docs:
        doc.metadata = {"source": specific_book}
        documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    print("\n--- Creating embeddings ---")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    print("\n--- Finished creating embeddings ---")

    print("\n--- Creating and persisting vector store ---")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(persistent_directory)  
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
