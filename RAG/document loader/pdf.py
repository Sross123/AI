from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document loader/⚛️ Master React with 300 Handpicked Questions!.pdf")

docs = data.load()
# print("count: ",len(docs))
splitter = RecursiveCharacterTextSplitter(
     chunk_size = 10000,
     chunk_overlap=10
 )

chunks = splitter.split_documents(docs)


print("\n\n")
print(chunks[0].page_content)