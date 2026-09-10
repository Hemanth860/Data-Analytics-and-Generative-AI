"""
FastAPI Application for Zepto GenAI Support Assistant (/support_assistant/main.py)
Author: Kammari Hemanth Kumar Achari

Provides REST API endpoints:
- POST /ask: Accepts {"query": str} and returns PolicyResponse JSON schema.
- GET /health: Health check endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag_engine import run_support_assistant, PolicyResponse

app = FastAPI(
    title="Zepto GenAI Policy Support Assistant",
    description="Grounded GenAI RAG service answering Zepto customer policy questions.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is Zepto's refund policy for damaged grocery items?")

@app.get("/")
def read_root():
    return {
        "service": "Zepto GenAI Support Assistant API",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=PolicyResponse)
def ask_question(request: QueryRequest):
    """
    POST /ask endpoint:
    Executes the LangGraph RAG flow and returns Pydantic validated response.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    try:
        response = run_support_assistant(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing RAG pipeline: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
