import os
from dotenv import load_dotenv
import faiss
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load the embedding model
embeddings = MistralAIEmbeddings(model="mistral-embed")

# Load the existing FAISS index from file
db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)


# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4}
)
relevant_docs = retriever.invoke("Odysseus")

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")