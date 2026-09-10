"""
Test FastAPI endpoints for Support Assistant (/support_assistant/test_fastapi.py)
Author: Kammari Hemanth Kumar Achari
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def run_tests():
    print("--- TESTING FASTAPI POST /ask ENDPOINT ---")
    
    # Test 1: Policy Question (Retrieval Triggered)
    query_policy = "What is Zepto refund policy for damaged or perishable items?"
    resp1 = client.post("/ask", json={"query": query_policy})
    print(f"\n[Test 1] Policy Query: '{query_policy}'")
    print(f"Status Code: {resp1.status_code}")
    json1 = resp1.json()
    print("Response JSON:\n", json.dumps(json1, indent=2))
    
    # Test 2: General Question (Direct Answer Triggered)
    query_general = "What is the capital of France?"
    resp2 = client.post("/ask", json={"query": query_general})
    print(f"\n[Test 2] General Query: '{query_general}'")
    print(f"Status Code: {resp2.status_code}")
    json2 = resp2.json()
    print("Response JSON:\n", json.dumps(json2, indent=2))
    
    # Save transcripts to file for README embedding
    transcripts = {
        "policy_query": {
            "request": {"query": query_policy},
            "response": json1
        },
        "general_query": {
            "request": {"query": query_general},
            "response": json2
        }
    }
    
    with open(os.path.join(os.path.dirname(__file__), "api_transcripts.json"), "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2)
        
    print("\nSaved api_transcripts.json successfully!")

if __name__ == "__main__":
    run_tests()
