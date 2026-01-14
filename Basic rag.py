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


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# 1. Initialize Gemini 2.0 Flash
# temperature=0 ensures facts, not stories (Critical for internships!)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# 2. Create the RAG Chain
# k=3 retrieves the top 3 most relevant facts
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3})
)

# 3. RUN IT
user_query = "What is the policy for remote work mentioned in the manual?"
result = qa_chain.invoke(user_query)

print("\n--- FINAL ANSWER ---")
print(result["result"])
