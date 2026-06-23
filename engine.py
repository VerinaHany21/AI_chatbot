import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load your .env file so os.getenv() can find your API key
load_dotenv()

# ─────────────────────────────────────────────
# SWITCH: Set to True to use Ollama locally
#         Set to False to use OpenRouter cloud
# ─────────────────────────────────────────────
USE_LOCAL_LLAMA = False

if USE_LOCAL_LLAMA:
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Ollama doesn't need a real key, but the library requires something
                          # you have to download Ollama and run a local model for this to work
    )
    # When using Ollama, we use a single local model (no fallback needed)
    MODELS = ["llama3.2:3b"]
    print("Using local Ollama model.")

else:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "RAG Chatbot"
        }
    )
    # Fallback list: if the first model is rate-limited, the next one is tried automatically
    MODELS = [
        "openrouter/auto",                          # auto-picks best available free model
        "meta-llama/llama-4-scout:free",            # fast, 128K context
        "meta-llama/llama-3.3-70b-instruct:free",  # powerful fallback
        "deepseek/deepseek-v3:free",               # strong general fallback
        "google/gemma-3-12b:free",                 # final fallback
    ]
    print("Using OpenRouter cloud model with fallback list.")

# Holds the uploaded PDF's vector data in memory
# Stays None until a PDF is uploaded
vector_store = None


def call_llm(messages, temperature=0.7):
    """
    Tries each model in MODELS list in order.
    If a model is rate-limited (429), it silently moves to the next one.
    If all models fail, returns a friendly error message.
    """
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                # This model is overloaded — try the next one
                print(f"{model} is rate limited, trying next model...")
                continue
            else:
                # A different error (bad key, network issue, etc.) — stop trying
                return f"Error: {error_str}"

    # Every model in the list failed
    return "All free models are currently busy. Please wait a moment and try again."


def process_pdf(pdf_file_path):
    """
    Takes a PDF file path, extracts the text, splits it into chunks,
    converts chunks to vectors, and stores them in ChromaDB.
    Called automatically when the user uploads a PDF in the UI.
    """
    global vector_store
    try:
        # Step 1: Extract text from every page of the PDF
        loader = PyPDFLoader(pdf_file_path)
        pages = loader.load()

        # Step 2: Split the text into overlapping chunks
        # chunk_size=1000: each chunk is up to 1000 characters
        # chunk_overlap=100: chunks share 100 characters with the next,
        #                    so sentences aren't cut off at boundaries
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        # Step 3: Load the free local embedding model (runs on your CPU, no API needed)
        # This converts each text chunk into a vector (a list of numbers representing meaning)
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Step 4: Store all vectors in ChromaDB (in memory — resets when app restarts)
        vector_store = Chroma.from_documents(chunks, embedder)

        print(f"PDF processed successfully: {len(chunks)} chunks stored.")
        return f"Success! Processed {len(chunks)} chunks from your PDF."

    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def route_and_generate(user_query, chat_history):
    """
    Main function called on every user message.
    1. Rebuilds conversation history
    2. Checks if a PDF is uploaded
    3. If yes: classifies the query (needs PDF or not?)
    4. Routes to either RAG pipeline or Generic pipeline
    5. Returns (reply_text, pipeline_label)
    """
    global vector_store

    # ── Rebuild conversation history ──────────────────────────────────────────
    # The LLM needs the full conversation so it can give contextual replies.
    # We strip the [RAG Pipeline] / [Generic Pipeline] labels from previous
    # assistant messages before sending them back — the LLM doesn't need those.
    messages = []
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            # Remove the label we added for display purposes
            content = content.split("]**\n\n")[-1] if "]**\n\n" in content else content
        messages.append({"role": role, "content": content})

    # ── No PDF uploaded: answer from general knowledge ────────────────────────
    if vector_store is None:
        messages.append({"role": "user", "content": user_query})
        reply = call_llm(messages, temperature=0.7)
        return reply, "Generic (no PDF uploaded)"

    # ── PDF is uploaded: classify the query first ─────────────────────────────
    # We ask the LLM: "does this question need the PDF or not?"
    # temperature=0.0 means no randomness — we need a clear RAG or GENERIC answer
    router_prompt = (
        f"Classify this query: '{user_query}'. "
        f"If it asks about a specific document, uploaded file, or custom text, reply ONLY with: RAG. "
        f"If it's general knowledge, coding help, or small talk, reply ONLY with: GENERIC."
    )

    try:
        intent = call_llm(
            [{"role": "user", "content": router_prompt}],
            temperature=0.0
        ).strip().upper()
    except Exception:
        intent = "RAG"  # If classification fails, default to RAG (safer choice)

    # ── RAG Pipeline ──────────────────────────────────────────────────────────
    if "RAG" in intent:
        # Search ChromaDB for the 5 chunks most similar to the user's question
        docs = vector_store.similarity_search(user_query, k=5)

        # Debug: print retrieved chunks to your terminal so you can verify
        print(f"\nRAG triggered. Retrieved {len(docs)} chunks:")
        for i, doc in enumerate(docs):
            print(f"  Chunk {i+1}: {doc.page_content[:120]}...")

        # Join all retrieved chunks into one context block
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build the system message — explicitly telling the LLM that
        # the context below IS the uploaded PDF content
        system_msg = (
            "You are a helpful assistant. The user has uploaded a PDF document. "
            "The following are extracted sections from that PDF — this IS the document content.\n\n"
            f"PDF Content:\n{context}\n\n"
            "Answer the user's question using this content. "
            "If the answer isn't present in the content above, say: "
            "'I couldn't find that in the uploaded PDF.'"
        )

        # Final message list: system context + conversation history + new question
        rag_messages = (
            [{"role": "system", "content": system_msg}]
            + messages
            + [{"role": "user", "content": user_query}]
        )

        reply = call_llm(rag_messages, temperature=0.1)
        return reply, "RAG Pipeline (answered from your PDF)"

    # ── Generic Pipeline ──────────────────────────────────────────────────────
    else:
        # No PDF context needed — just answer from the model's general knowledge
        messages.append({"role": "user", "content": user_query})
        reply = call_llm(messages, temperature=0.7)
        return reply, "Generic Pipeline (general knowledge)"