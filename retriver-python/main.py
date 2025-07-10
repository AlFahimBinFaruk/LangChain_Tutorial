from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import getpass
import os

load_dotenv()
file_path = "/home/bs00927/Downloads/nke-10k-2023.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

# print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# print(all_splits)

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")



vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
# ids = vector_store.add_documents(documents=all_splits)
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])
