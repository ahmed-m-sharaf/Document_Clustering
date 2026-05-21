FROM python:3.12-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Download the required spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the application source code and models
COPY app/ ./app/
COPY utils/ ./utils/
COPY models/ ./models/

# Make sure start.sh has executable permissions
RUN chmod +x app/start.sh

# Expose ports for FastAPI (8000) and Streamlit (7860 for Hugging Face)
EXPOSE 8000
EXPOSE 7860

# Run the applications via startup script
CMD ["./app/start.sh"]
