FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for opencv
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Install torch CPU-only first (much smaller than full torch)
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install rest of dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & sleep 5 && streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"]