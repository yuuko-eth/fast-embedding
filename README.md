# fast-embedding

FastEmbedding: A FastAPI server that wraps FlagEmbedding for both single and batch text embedding inference.

## Features

- Single text embedding endpoint
- Batch text embedding endpoint
- Configurable model via environment variables
- Health check and model info endpoints
- CUDA support with manual PyTorch installation or compute backend of your choice
- Request/response validation with Pydantic
- Comprehensive error handling and logging

## Setup

### 1. Install Dependencies

Using pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Using uv (automatically creates venv at `.venv`):

```bash
uv sync
```

### 2. Install PyTorch with CUDA Support

Since PyTorch installation depends on your CUDA version, install it manually first:

```bash
# Activate venv first
python3 -m venv .venv

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Configure Environment

Copy the example environment file and modify as needed:

```bash
cp .env.example .env
```

Edit `.env` to set your desired model and configuration:

```bash
MODEL_ID=BAAI/bge-m3  # Default BGE-M3 model
HOST=0.0.0.0
PORT=8000
CUDA_AVAILABLE=true
```

### 4. Run the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Single Text Embedding
```bash
POST /embed
Content-Type: application/json

{
    "text": "Your text to embed",
    "instruction": "Optional instruction"  # Optional
}
```

### Batch Text Embedding
```bash
POST /embed/batch
Content-Type: application/json

{
    "texts": ["First text", "Second text", "Third text"],
    "instruction": "Optional instruction"  # Optional
}
```

### Model Information
```bash
GET /model/info
```

## Usage Examples

### Python Client Example

```python
import requests
import json

base_url = "http://localhost:8000"

# Single embedding
response = requests.post(f"{base_url}/embed", json={
    "text": "This is a sample text for embedding"
})
embedding_data = response.json()
print(f"Embedding dimension: {embedding_data['dimension']}")
print(f"Embedding vector: {embedding_data['embedding'][:5]}...")  # First 5 elements

# Batch embedding
response = requests.post(f"{base_url}/embed/batch", json={
    "texts": [
        "First document to embed",
        "Second document to embed", 
        "Third document to embed"
    ]
})
batch_data = response.json()
print(f"Number of embeddings: {batch_data['count']}")
print(f"Embedding dimension: {batch_data['dimension']}")
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Single embedding
curl -X POST "http://localhost:8000/embed" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, world!"}'

# Batch embedding
curl -X POST "http://localhost:8000/embed/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'

# Model info
curl -X GET "http://localhost:8000/model/info"
```

## Supported Models

The server supports any FlagEmbedding model. Popular choices include:

- `BAAI/bge-m3` (default) - Multilingual, supports dense, sparse, and multi-vector retrieval
- `BAAI/bge-large-en-v1.5` - Large English model
- `BAAI/bge-base-en-v1.5` - Base English model  
- `BAAI/bge-small-en-v1.5` - Small English model

Set the `MODEL_ID` environment variable to use a different model.

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL_ID` | `BAAI/bge-m3` | HuggingFace model identifier |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `CUDA_AVAILABLE` | `true` | Whether to use CUDA if available |
| `RELOAD` | `false` | Enable auto-reload for development |
| `LOG_LEVEL` | `info` | Logging level |

## Performance Notes

- The server uses FP16 (half precision) by default for better performance on GPU
- Batch embedding is more efficient than multiple single requests
- Maximum batch size is limited to 100 texts to prevent memory issues
- First request may be slower due to model loading and CUDA initialization

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad request (invalid input)
- `500` - Internal server error (model/processing error)
- `503` - Service unavailable (model not loaded)

Error responses include detailed error messages in the response body.

## Docker Support

### 1. Building the image

```bash
docker build -t fast-embedding-server .
```

### 2. Run the image

Make sure to supply `--gpus all` or numerical index to assign compute device:

```bash
docker run --gpus=all -d --restart=always fast-embedding-server
```
