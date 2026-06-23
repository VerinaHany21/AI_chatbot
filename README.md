---
title: AI Chatbot
emoji: 🤖
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
python_version: '3.13'
app_file: app.py
pinned: false
short_description: Conversational AI chatbot with RAG pipeline
---

# AI Chatbot with LLM & RAG

A conversational AI assistant that answers questions from general knowledge **and** from uploaded PDF documents. Built with Python, LangChain, ChromaDB, and Gradio — runs fully offline via Ollama or in the cloud via OpenRouter.

---

## How It Works

```
User sends a message
        ↓
Is a PDF uploaded?
   ├── No  → Generic Pipeline (general knowledge)
   └── Yes → Route query via LLM classifier
                 ├── GENERIC → answer from general knowledge
                 └── RAG     → retrieve PDF chunks → inject context → answer from PDF
```

**RAG Pipeline steps:**
1. Uploaded PDF is split into overlapping 1000-character chunks
2. Each chunk is embedded using `all-MiniLM-L6-v2` (runs locally, no API needed)
3. Vectors are stored in an in-memory ChromaDB instance
4. On each query, the 5 most semantically similar chunks are retrieved
5. Retrieved chunks are injected into the system prompt; the LLM answers from them only

**Query routing** uses a separate LLM call at `temperature=0.0` to classify whether the user's question targets the uploaded document or general knowledge — keeping the two pipelines cleanly separated.

---

## Key Features

- **Dual deployment modes** — toggle between local Ollama and OpenRouter cloud with a single boolean flag (`USE_LOCAL_LLAMA` in `engine.py`)
- **Automatic model fallback** — if a cloud model returns a 429 rate-limit error, the next model in the priority list is tried silently
- **PDF ingestion** — chunking with 100-character overlap prevents sentences from being cut off at boundaries
- **Pipeline transparency** — every reply is labelled `[RAG Pipeline]` or `[Generic Pipeline]` so the routing decision is always visible

---

## Project Structure

```
ai-chatbot/
├── app.py              # Gradio UI
├── engine.py           # LLM calls, RAG pipeline, PDF processing, query router
├── requirements.txt    # Python dependencies
├── .env                # API key (never commit this)
└── .gitignore
```

---

## Live Deployment

🔗 **[Live Demo](https://huggingface.co/spaces/verinahany/AI_chatbot)**

---

## Setup

### Prerequisites

- Python 3.9+
- Git
- Ollama (only for local mode)

### Install dependencies

```bash
git clone https://github.com/VerinaHany21/AI_chatbot.git
cd AI_chatbot
python -m venv venv

# Activate — Windows:
venv\Scripts\activate
# Activate — Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Local mode (offline, free)

1. Download and install [Ollama](https://ollama.com)
2. Pull the model: `ollama pull llama3.2:3b`
3. Set `USE_LOCAL_LLAMA = True` in `engine.py`
4. In a separate terminal: `ollama serve`
5. Run: `python app.py`

### Cloud mode (OpenRouter)

1. Create a free account at [openrouter.ai](https://openrouter.ai) and generate an API key
2. Create a `.env` file in the project root:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   ```
3. Set `USE_LOCAL_LLAMA = False` in `engine.py`
4. Run: `python app.py`

The app is available at `http://127.0.0.1:7860`.

---

## Usage

| Task | What to do |
|---|---|
| General question | Type and send — no PDF needed |
| PDF question | Upload a PDF, wait for *"Success! Processed X chunks"*, then ask |
| Clear history | Click **Clear chat** |

---

## Cloud Model Fallback Order

When running in cloud mode, models are tried in this order if any is rate-limited:

1. `openrouter/auto` (auto-selects best available free model)
2. `meta-llama/llama-4-scout:free`
3. `meta-llama/llama-3.3-70b-instruct:free`
4. `deepseek/deepseek-v3:free`
5. `google/gemma-3-12b:free`

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM (local) | Ollama + Llama 3.2 3B |
| LLM (cloud) | OpenRouter API |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector store | ChromaDB (in-memory) |
| RAG framework | LangChain |
| PDF parsing | PyPDF |
| UI | Gradio |
| Environment | python-dotenv |

---

## Requirements

```
openai
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
chromadb
gradio
python-dotenv
pypdf
sentence-transformers
```

Install with: `pip install -r requirements.txt`