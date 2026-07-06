from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

file_path = Path(__file__).parent / "document loader/⚛️ Master React with 300 Handpicked Questions!.pdf"


data = PyPDFLoader(str(file_path))

docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarizes the text"),
    ("human", "{data}")]
)

docs = data.load()
model = ChatMistralAI(model_name="mistral-small-2603")

prompt = template.format_messages(data=docs)

response = model.invoke(prompt)

print("\n\n")

print(response.content)