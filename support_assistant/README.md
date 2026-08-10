# Module 3: GenAI Policy Support Assistant (`/support_assistant`)

**Module Marks:** 25 Marks  
**Author:** AI/ML Engineer (B.Tech Capstone Project)  
**Target Domain:** Grounded RAG (Retrieval-Augmented Generation) & FastAPI Service  

---

## 📌 Executive Summary
This module implements an end-to-end, grounded **GenAI Support Assistant** for Zepto. It ingests 8 official Zepto policy documents, indexes them into a local **ChromaDB** vector database using `all-MiniLM-L6-v2` embeddings, orchestrates intent routing & retrieval via a **LangGraph StateGraph**, enforces a **Pydantic output schema**, and serves the service over a **FastAPI** REST endpoint.

Per task specifications:
* **Graded Baseline (`MOCK_LLM=1` or unset):** Runs in a 100% deterministic, offline mock mode requiring zero API keys, no signup, and no network calls to LLM providers.
* **Optional Extension (`MOCK_LLM=0`):** Toggles real LLM inference via free-tier APIs (e.g. Groq API) using the structured prompt template in `prompts.py`.

---

## 🏗️ RAG Pipeline Architecture & Component Walkthrough

```
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  Zepto Corpus   │ ────> │  Embedding Engine      │ ────> │  ChromaDB Vector DB  │
│  (8 Policy Docs)│       │ (all-MiniLM-L6-v2)     │       │ ('zepto_policies')   │
└─────────────────┘       └────────────────────────┘       └──────────┬───────────┘
                                                                      │
                                                                      ▼
┌──────────────────┐      ┌────────────────────────┐       ┌──────────────────────┐
│ Customer Query   │ ────>│ Node 1: classify_intent│ ─────>│ Node 2: retrieve_    │
│ (POST /ask API)  │      │ (Keyword Heuristic)    │       │         and_answer   │
└──────────────────┘      └───────────┬────────────┘       └──────────┬───────────┘
                                      │                               │
                                      ▼                               ▼
                          ┌────────────────────────┐       ┌──────────────────────┐
                          │ Node 3: direct_answer  │       │ Pydantic Validation  │
                          │ (General Query Fallback│       │ (PolicyResponse)     │
                          └────────────────────────┘       └──────────────────────┘
```

### Stage-by-Stage Flow:
1. **Ingestion Stage (`ingest_policies.py`):** Reads the 8 official Zepto policy text files (`docs/doc_01.txt` ... `docs/doc_08.txt`).
2. **Embedding Stage (`ingest_policies.py`):** Embeds each document chunk using open-source `sentence-transformers` (`all-MiniLM-L6-v2`).
3. **Vector Storage Stage (`ingest_policies.py`):** Stores embeddings and document metadata into a local persistent **ChromaDB** collection (`zepto_policies`).
4. **Intent Classification Node (`rag_engine.py` -> `classify_intent_node`):**
   * Inspects incoming query using a keyword heuristic (`["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]`).
   * Classifies query as `policy_question` or `general_question`.
5. **Retrieval & Answer Node (`rag_engine.py` -> `retrieve_and_answer_node`):**
   * For `policy_question`: Performs real vector cosine similarity retrieval against ChromaDB for top-3 chunks.
   * **Mock Mode (`MOCK_LLM=1`):** Formats grounded answer as `f"Based on the retrieved context: {top_chunk_snippet}"`, sets `sources = [doc_ids]`, `confidence = 1.0`.
   * **Real LLM Mode (`MOCK_LLM=0`):** Prompts LLM using structured template in `prompts.py`.
6. **Direct Answer Node (`rag_engine.py` -> `direct_answer_node`):**
   * For `general_question`: Returns canned string `"I can only answer questions about Zepto policies right now."`, `sources = []`, `confidence = 1.0`.
7. **Pydantic Validation & API Output (`main.py`):** Enforces `PolicyResponse(answer, sources, confidence)` schema.

---

## 📜 Executed API Call Transcripts (`POST /ask`)

### Test 1: Policy Question Query (Retrieval Triggered)
**Request:**
```json
POST /ask
{
  "query": "What is Zepto refund policy for damaged or perishable items?"
}
```

**Response (HTTP 200 OK):**
```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_05"
  ],
  "confidence": 1.0
}
```

---

### Test 2: General Question Query (Direct Answer Triggered)
**Request:**
```json
POST /ask
{
  "query": "What is the capital of France?"
}
```

**Response (HTTP 200 OK):**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## 🐳 Docker Deployment Instructions

### Build Docker Image:
```bash
docker build -t zepto-support-assistant support_assistant/
```

### Run Container Locally (Port 7860):
```bash
docker run -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

### Access Local API:
* OpenAPI Documentation: `http://localhost:7860/docs`
* POST `/ask` Endpoint: `http://localhost:7860/ask`

---

## 🚀 Execution Commands

### Run FastAPI Service locally via Uvicorn:
```bash
python support_assistant/main.py
```

### Run Interactive Streamlit Dashboard:
```bash
streamlit run support_assistant/ui.py
```
