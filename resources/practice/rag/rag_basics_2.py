import os
from dotenv import load_dotenv
import faiss
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "faiss_db_for_basics")

embeddings = MistralAIEmbeddings(model="mistral-embed")

db = FAISS.load_local(persistent_directory, embeddings, allow_dangerous_deserialization=True)


retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4}
)
relevant_docs = retriever.invoke("odyssey")

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")