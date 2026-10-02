# Retrieval-Augmented Generation (RAG) Q&A System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

A question-answering assistant built with **LangChain**, **OpenAI**, and **ChromaDB** that answers questions about the **Eleven Madison Park** restaurant using text collected from its website. Answers are grounded in retrieved document chunks and returned together with their sources, through an interactive **Gradio** web interface.

---

## Overview

Large language models can answer general questions, but they don't know the details of a specific business. This project uses **Retrieval-Augmented Generation (RAG)** to fix that: the restaurant's website text is split into chunks, embedded into a vector store, and the most relevant chunks are retrieved and passed to the LLM as context for each question.

## Key Features

- Loads a text knowledge base (`eleven_madison_park_data.txt`) with LangChain's `TextLoader`
- Splits documents into overlapping chunks with `RecursiveCharacterTextSplitter`
- Creates vector embeddings with `OpenAIEmbeddings` and stores them in **ChromaDB**
- Tests semantic similarity search directly against the vector store
- Answers questions with a `RetrievalQAWithSourcesChain`, returning both the **answer** and its **sources**
- Interactive **Gradio** app with a question box, answer/source outputs, a clear button, and example questions
- API key loaded from a `.env` file via `python-dotenv`

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| LLM Framework | LangChain (`langchain`, `langchain-openai`, `langchain-community`) |
| LLM & Embeddings | OpenAI API (`ChatOpenAI`, `OpenAIEmbeddings`) |
| Vector Database | ChromaDB |
| Tokenization | tiktoken |
| User Interface | Gradio |
| Configuration | python-dotenv |
| Environment | Jupyter Notebook |

## RAG Workflow

```
 Website text data (.txt)
          │
          ▼
 ┌─────────────────┐
 │  Load document  │  TextLoader
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Split chunks   │  RecursiveCharacterTextSplitter (size 1000, overlap 150)
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Embed & store  │  OpenAIEmbeddings → ChromaDB
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │    Retrieve     │  Top-3 most similar chunks for the user's question
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │    Generate     │  ChatOpenAI (temperature 0) + "stuff" chain
 └────────┬────────┘
          ▼
   Answer + Sources  →  Gradio interface
```

## How It Works

1. **Setup** – Libraries are installed and the OpenAI API key is loaded from `.env`.
2. **Data loading** – The Eleven Madison Park text file is loaded as a LangChain document. Each section of the file contains a source URL, a page title, and the page content.
3. **Chunking** – The document is split into chunks of 1,000 characters with a 150-character overlap so context is preserved across chunk boundaries.
4. **Embeddings & vector store** – Each chunk is converted into a 1,536-dimensional embedding and stored in an in-memory ChromaDB collection.
5. **Retrieval test** – A similarity search (e.g. *"Is there a dress code?"*) verifies that relevant chunks are returned.
6. **RAG chain** – A retriever (`k=3`) is combined with `ChatOpenAI` in a `RetrievalQAWithSourcesChain` that uses the `"stuff"` strategy to place the retrieved chunks into the prompt.
7. **Gradio app** – The `ask_elevenmadison_assistant` function sends the user's question through the chain and displays the answer and its sources in a web UI.

### Example

**Question:** *What kind of food does Eleven Madison Park serve?*

**Answer:** *Eleven Madison Park serves a fully plant-based menu, using no animal products.*

## Getting Started

### Prerequisites

- Python 3.11 (the version the notebook was run with)
- An [OpenAI API key](https://platform.openai.com/api-keys)
- The data file `eleven_madison_park_data.txt` (not included in this repository)

### Installation

```bash
# Clone the repository
git clone https://github.com/Mr-Sari/Retrieval-Augmented-Generation-RAG-Q-A-System.git
cd Retrieval-Augmented-Generation-RAG-Q-A-System

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy `.env.example` to `.env` and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. Place `eleven_madison_park_data.txt` in the `data/` folder and update `DATA_FILE_PATH` in the notebook to point to it, for example:

   ```python
   DATA_FILE_PATH = "data/eleven_madison_park_data.txt"
   ```

### Run the Notebook

```bash
jupyter notebook RAG_LangChain.ipynb
```

Run the cells from top to bottom. The final cell launches the Gradio app, available locally at the URL printed in the output (e.g. `http://127.0.0.1:7860`).

## Project Structure

```
Retrieval-Augmented-Generation-RAG-Q-A-System/
├── data/                   # Place eleven_madison_park_data.txt here
├── RAG_LangChain.ipynb     # Main notebook: loading, chunking, embeddings, RAG chain, Gradio app
├── requirements.txt        # Python dependencies
├── .env.example            # Template for the OpenAI API key
├── .gitignore
└── README.md
```

## Notes

- The vector store is created **in memory**, so embeddings are rebuilt each time the notebook runs.
- Running the notebook calls the OpenAI API (embeddings + chat completions), which uses your API credits.
- The notebook uses LangChain's legacy import paths, so `requirements.txt` pins `langchain<1.0`.
- Never commit your `.env` file — it is already listed in `.gitignore`.
