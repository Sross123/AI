#load pdf
#split into chunks
#create the embaddings
#store into chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai.embeddings import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document loader/master_react.pdf")
docs = data.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

embadding_modal = MistralAIEmbeddings()

vectorStore = Chroma.from_documents(
    documents=chunks,
    embedding=MistralAIEmbeddings(),
    persist_directory="chroma_db"
)

