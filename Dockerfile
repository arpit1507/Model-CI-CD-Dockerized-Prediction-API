# Simple Dockerfile for the FastAPI model server
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src/ src/
COPY models/ models/

# Expose port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]