# RAG Practice

A simple Retrieval-Augmented Generation (RAG) project that lets you chat with a PDF document using Chroma and Mistral.

## Features

- Loads a PDF from the document folder
- Splits the document into chunks
- Stores embeddings in a local Chroma database
- Chat with the document through a Streamlit UI

## Setup

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   On Windows PowerShell:
   ```powershell
   .venv\\Scripts\\Activate.ps1
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set your Mistral API key
   ```bash
   export MISTRAL_API_KEY="your_key_here"
   ```
   On Windows PowerShell:
   ```powershell
   $env:MISTRAL_API_KEY="your_key_here"
   ```

## Build the vector database

Run the indexing script:

```bash
python create_db.py
```

## Run the chat app

```bash
streamlit run main.py
```

## Project files

- create_db.py: indexes the PDF into Chroma
- main.py: Streamlit chat interface for querying the indexed document
- document/: folder containing the PDF
- chroma_db/: local vector database storage
