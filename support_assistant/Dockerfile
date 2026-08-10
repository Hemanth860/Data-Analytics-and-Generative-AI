# Dockerfile for Zepto GenAI Policy Support Assistant
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt-lists/*

# Copy project files
COPY . /app/

# Install python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic sentence-transformers chromadb langgraph

# Expose FastAPI service port
EXPOSE 7860

# Set environment variable default (MOCK_LLM=1 for graded baseline)
ENV MOCK_LLM=1

# Entry point command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
