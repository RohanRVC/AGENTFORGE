# -----------------------------
# Base Python Image
# -----------------------------
    FROM python:3.11-slim

    # Disable Python buffering
    ENV PYTHONUNBUFFERED=1
    
    # Set work directory inside the container
    WORKDIR /app
    
    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        git \
        build-essential \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy project files into container
    COPY . /app
    
    # Install Python dependencies
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Expose API port
    EXPOSE 8000
    
    # Start FastAPI server
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]