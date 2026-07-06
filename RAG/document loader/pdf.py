from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loader/⚛️ Master React with 300 Handpicked Questions!.pdf")

docs = data.load()
print("count: ",len(docs))
print("\n\n")
print(docs)