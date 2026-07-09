from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

file_path = Path(__file__).parent / "document loader/deeplearning.pdf"


data = PyPDFLoader(str(file_path))

docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarizes the text"),
    ("human", "{data}")]
)

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
model = ChatMistralAI(model_name="mistral-small-2603")

prompt = template.format_messages(data=chunks)

response = model.invoke(prompt)

print("\n\n")

print(response.content)