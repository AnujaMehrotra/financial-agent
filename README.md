📊 Multi-Agent Financial Analysis with LangGraph & LangChain
This project is a modular, multi-agent system built using LangGraph, LangChain, and OpenAI APIs. It processes and analyzes financial filings (e.g., 10-Ks) using Retrieval-Augmented Generation (RAG) and task-specific agents.

🚀 Features
Retriever Agent – Fetches relevant document chunks via FAISS-based vector search.

Validator Agent – Filters irrelevant or misaligned content using LLM-based strict validation.

Summarizer Agent – Creates a concise, context-aware summary from validated documents.

Analyst Agent – Performs deeper analytical tasks.

User Agent – Accepts user queries.

Graph Orchestration – Manages execution flow using LangGraph.

🗂️ Project Structure
src/
├── agents/               # Modular agents (retriever, summarizer, validator, etc.)
├── retrieval/            # FAISS-based RAG retrieval logic
├── utils/                # Config and loaders
├── validation/           # Validation registry (optional)
├── main.py               # Entry point
├── graph_builder.py      # LangGraph flow definition
data/
└── tesla_10k_2023.txt    # Sample input file
🧩 Requirements
Python 3.10+

OpenAI API Key

Install dependencies:
pip install -r requirements.txt

🔑 Setup
Create a .env file or export your API key:
export OPENAI_API_KEY=your-key-here

Place your financial text file in data/ folder. Example:
data/tesla_10k_2023.txt

Run the pipeline:
python src/main.py

✨ Example Query
input_state = {
    "question": "What are Tesla's key business risks mentioned in 2023?",
}

📌 TODO
 Add CLI support

 Extend validator with LangChain guardrails

 Include Pinecone or Chroma for scalable RAG

📄 License
MIT License
