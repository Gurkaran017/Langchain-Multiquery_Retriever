from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from documents import get_documents
import config

def get_vector_store():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docs = get_documents()
    
    return FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)