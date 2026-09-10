"""
Policy Ingestion & Vector Indexing Pipeline (/support_assistant/ingest_policies.py)
Author: Kammari Hemanth Kumar Achari

This module loads Zepto's 8 official policy documents, embeds them using 
open-source sentence-transformers (all-MiniLM-L6-v2), and indexes them into 
a local ChromaDB vector store collection ('zepto_policies').
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "zepto_policies"

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model '{MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return collection

def ingest_documents():
    """Load, chunk, embed, and index all 8 Zepto policy document files into ChromaDB."""
    print("Starting document ingestion into ChromaDB...")
    collection = get_chroma_collection()
    
    # Reset existing documents if present
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection '{COLLECTION_NAME}' already has {existing_count} records.")
        return collection
        
    model = get_embedding_model()
    
    doc_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
    doc_files.sort()
    
    ids = []
    documents = []
    metadatas = []
    
    for filename in doc_files:
        filepath = os.path.join(DOCS_DIR, filename)
        doc_id = filename.replace(".txt", "")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        ids.append(doc_id)
        documents.append(content)
        metadatas.append({"source": filename, "doc_id": doc_id})
        
    print(f"Generating embeddings for {len(documents)} document chunks...")
    embeddings = model.encode(documents).tolist()
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"Successfully indexed {len(documents)} documents into ChromaDB collection '{COLLECTION_NAME}'.")
    return collection

def query_policy_context(query, top_k=3):
    """Embed query and retrieve top_k most similar document chunks via cosine similarity."""
    collection = get_chroma_collection()
    model = get_embedding_model()
    
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    retrieved_chunks = []
    if results and "documents" in results and len(results["documents"]) > 0:
        docs = results["documents"][0]
        ids = results["ids"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0] if "distances" in results else [0.0]*len(docs)
        
        for doc, doc_id, meta, dist in zip(docs, ids, metas, dists):
            retrieved_chunks.append({
                "doc_id": doc_id,
                "content": doc,
                "metadata": meta,
                "distance": dist
            })
            
    return retrieved_chunks

if __name__ == "__main__":
    ingest_documents()
    res = query_policy_context("What is Zepto refund policy?")
    print(f"Test Retrieval Top Result Doc ID: {res[0]['doc_id']}")
    print(f"Snippet: {res[0]['content'][:150]}...")
