#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Document Clustering Application..."

# Start FastAPI in the background
echo "Starting FastAPI backend on port 8000..."
uvicorn app.api:app --host 0.0.0.0 --port 8000 &

# Wait for a moment to let FastAPI start
sleep 3

# Get port from environment or default to 7860 for Hugging Face Spaces
PORT=${PORT:-7860}
echo "Starting Streamlit frontend on port $PORT..."
streamlit run app/ui.py --server.port $PORT --server.address 0.0.0.0

