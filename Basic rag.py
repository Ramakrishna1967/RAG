import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader("your_internship_data.pdf")
raw_data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150
)
chunks = text_splitter.split_documents(raw_data)

print(f"Block 1 Complete: Created {len(chunks)} chunks.")

