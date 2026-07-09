from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

file_path = Path(__file__).parent / "notes.txt"

data = TextLoader(str(file_path))

docs = data.load()

chunks = splitter.split_documents(docs)

# print(docs[0].metadata["source"])
# print(chunks)

for i in chunks:
    print()
    print(i.page_content)
    print()
    print()
    print()
