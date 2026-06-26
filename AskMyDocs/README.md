# AskMyDocs

## Overview

AskMyDocs is a local Retrieval-Augmented Generation (RAG) prototype for question answering over PDF documents. The workflow is implemented in a Jupyter notebook and demonstrates how to load PDFs, split them into chunks, create embeddings, store them in a local Chroma vector database, and query them with an Ollama-based language model.

## What the project does

- Loads PDF files from the `my_pdfs/` folder using `PyPDFDirectoryLoader`
- Splits each document into smaller chunks with `RecursiveCharacterTextSplitter`
- Generates embeddings with `HuggingFaceEmbeddings`
- Stores and retrieves chunks from a local Chroma vector store
- Builds a prompt from the retrieved context and uses Ollama to generate an answer

## Project structure

- `AskMyDocs.ipynb`: the main notebook containing the implementation and example queries
- `my_pdfs/`: input folder for source PDFs (currently contains `alice.pdf`)
- `chroma/`: local Chroma database folder and persisted index files

## Requirements

The notebook uses these Python packages:

- `langchain-community`
- `langchain-core`
- `langchain-text-splitters`
- `chromadb`
- `pypdf`
- `sentence-transformers`
- `torch`

Install them with:

```bash
pip install langchain-community langchain-core langchain-text-splitters chromadb pypdf sentence-transformers torch
```

You also need a working Ollama installation and the model used by the notebook:

```bash
ollama pull gemma4:e2b
```

Make sure the Ollama service is running before executing the notebook.

## Configuration

The notebook uses the following local settings:

- `CHROMA_PATH = "chroma"`
- `DATA_PATH = "my_pdfs"`
- `model_path = r"D:\AI\MODELS\all-MiniLM-L6-v2"`

Update `model_path` to a valid local sentence-transformers model directory on your machine.

## How it works

1. `load_documents()` reads all PDF files from `my_pdfs/`.
2. `split_documents()` breaks each document into smaller chunks.
3. `calculate_chunk_ids()` assigns a unique ID to each chunk.
4. `add_to_chroma()` stores the chunks in the local Chroma database.
5. `query_rag()` performs similarity search, builds a prompt from the retrieved context, and calls Ollama for the final answer.

## Usage

1. Put one or more PDFs inside `my_pdfs/`.
2. Open `AskMyDocs.ipynb`.
3. Update the embedding model path in the notebook to match your local setup.
4. Run the notebook cells in order:
   - imports and configuration
   - `load_documents()` and `split_documents()`
   - `add_to_chroma(chunks)`
   - `query_rag("your question here")`

To rebuild the vector store from scratch, run `clear_database()` before re-adding documents.

## Notes

- The repository is intended as a local demo/prototype and may need tuning for larger or more diverse document collections.

## Author

- Developed by: Omar Hafez Khalil
- GitHub: [OmarHKhalil](https://github.com/OmarHKhalil)
- LinkedIn: [Omar Khalil](https://www.linkedin.com/in/omar-khalil-55a674281)
