---
title: Hybrid AI Chatbot with RAG
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Hybrid AI Chatbot with RAG

An intelligent, production-ready conversational assistant built with Python, Gradio, and LangChain. This project features a dual-pipeline routing architecture that seamlessly toggles between local offline execution (via Ollama) and robust cloud endpoints (via OpenRouter), incorporating Retrieval-Augmented Generation (RAG) to dynamically query custom PDF documentation.

## 🧠 System Architecture (How it Works)

This application is designed to mimic enterprise-level AI systems by decoupling the hardware layer from the logic layer. 

1. **The Gateway Switch:** The core engine contains a master boolean toggle (`USE_LOCAL_LLAMA`). When set to `True`, the system routes all traffic to a local, hardware-bound model (e.g., LLaMA 3.2 3B). When `False`, it routes to the OpenRouter API cloud infrastructure.
2. **Dynamic Intent Router:** Every user prompt is intercepted by a "Gatekeeper" LLM prompt set to `temperature=0.0`. This router evaluates the syntax and intent of the user's text to determine if they are asking a general knowledge question, or if they are referencing an uploaded document.
3. **The RAG Pipeline:** If the router detects a document reference, the system queries a local **ChromaDB** vector database. It uses `all-MiniLM-L6-v2` (via Hugging Face) to find the text fragments most semantically similar to the user's question, injects them into a strict context window, and forces the LLM to answer using *only* the retrieved facts.
4. **The Generic Pipeline:** If no document is needed, the system bypasses the vector database entirely and acts as a standard conversational agent at `temperature=0.7` for fluid, natural dialogue.

## ✨ Key Features

* **Hardware Agnostic:** Can run completely offline on edge devices or scale infinitely using cloud API endpoints.
* **Resilient Cloud Failover:** Built-in sequential fallback orchestration automatically cycles through a priority list of free cloud LLMs (Llama 3.3, DeepSeek, Gemma) if upstream providers throw `429 Rate Limit` exceptions.
* **Automated PDF Ingestion:** Uses `PyPDFLoader` and a `RecursiveCharacterTextSplitter` to handle document chunking with logical sentence overlaps, preventing data loss at fragment boundaries.
* **Modern API UI Stream:** Utilizes Gradio's modern dictionary-based chat history format, cleanly managing system/user/assistant roles natively.

---

## 🛠️ Local Installation & Setup (For Developers)

If you are a developer looking to review the code or run this architecture locally on your machine, follow these steps:

### 1. Prerequisites
* **Python 3.9+**
* **Ollama** (Only required if you intend to test the local hardware execution path).

### 2. Clone and Install Dependencies
```bash
git clone [https://github.com/VerinaHany21/AI_chatbot.git]
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt