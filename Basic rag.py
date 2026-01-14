import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader("your_data.pdf")
raw_data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150
)
chunks = text_splitter.split_documents(raw_data)

print(f"Block 1 Complete: Created {len(chunks)} chunks.")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

os.environ["GOOGLE_API_KEY"] = "YOUR_KEY_HERE"

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vector_db = FAISS.from_documents(chunks, embeddings)

print("Block 2 Complete: Vector Database ready.")
