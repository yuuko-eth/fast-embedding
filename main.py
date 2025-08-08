from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import os
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the model
model = None

class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Text to embed")
    instruction: Optional[str] = Field(None, description="Optional instruction for embedding")

class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")
    instruction: Optional[str] = Field(None, description="Optional instruction for embedding")

class EmbeddingResponse(BaseModel):
    embedding: List[float] = Field(..., description="Text embedding vector")
    dimension: int = Field(..., description="Embedding dimension")

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Embedding dimension")
    count: int = Field(..., description="Number of embeddings")

class HealthResponse(BaseModel):
    status: str
    model_id: str
    embedding_dimension: Optional[int] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    try:
        from FlagEmbedding import BGEM3FlagModel
        
        model_id = os.getenv("MODEL_ID", "BAAI/bge-m3")
        logger.info(f"Loading model: {model_id}")
        
        # Initialize the model
        model = BGEM3FlagModel(
            model_name_or_path=model_id,
            use_fp16=False,  # Use half precision for better performance
            device="cuda" if os.getenv("CUDA_AVAILABLE", "true").lower() == "true" else "cpu"
        )
        
        logger.info(f"Model {model_id} loaded successfully")
        for param in model.model.parameters():
            print(param.dtype)
            break
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="FlagEmbedding API Server",
    description="FastAPI server for FlagEmbedding with single and batch inference",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "FlagEmbedding API Server",
        "version": "1.0.0",
        "model_id": os.getenv("MODEL_ID", "BAAI/bge-m3")
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Test with a simple embedding to verify model is working
        test_embedding = model.encode("test", max_length=512)
        dimension = len(test_embedding['dense_vecs']) if isinstance(test_embedding, dict) else len(test_embedding)
        
        return HealthResponse(
            status="healthy",
            model_id=os.getenv("MODEL_ID", "BAAI/bge-m3"),
            embedding_dimension=dimension
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Model health check failed: {str(e)}")

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_single_text(request: EmbeddingRequest):
    """Embed a single text"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare text with instruction if provided
        text_to_embed = f"{request.instruction} {request.text}" if request.instruction else request.text
        
        # Generate embedding
        embedding_result = model.encode(text_to_embed, max_length=8192)
        
        # Extract dense vector (BGE-M3 returns dict with 'dense_vecs', 'sparse_vecs', etc.)
        if isinstance(embedding_result, dict):
            embedding = embedding_result['dense_vecs'].tolist()
        else:
            embedding = embedding_result.tolist()
        
        return EmbeddingResponse(
            embedding=embedding,
            dimension=len(embedding)
        )
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def embed_batch_texts(request: BatchEmbeddingRequest):
    """Embed multiple texts in batch"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(request.texts) > 100:  # Reasonable batch size limit
        raise HTTPException(status_code=400, detail="Batch size too large (max 100 texts)")
    
    try:
        # Prepare texts with instruction if provided
        texts_to_embed = [
            f"{request.instruction} {text}" if request.instruction else text 
            for text in request.texts
        ]
        
        # Generate embeddings for batch
        embedding_results = model.encode(texts_to_embed, max_length=8192)
        
        # Extract dense vectors
        if isinstance(embedding_results, dict):
            embeddings = [emb.tolist() for emb in embedding_results['dense_vecs']]
        else:
            embeddings = [emb.tolist() for emb in embedding_results]
        
        dimension = len(embeddings[0]) if embeddings else 0
        
        return BatchEmbeddingResponse(
            embeddings=embeddings,
            dimension=dimension,
            count=len(embeddings)
        )
        
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch embedding generation failed: {str(e)}")

@app.get("/model/info", response_model=dict)
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get a sample embedding to determine dimensions
        sample_embedding = model.encode("sample text", max_length=512)
        
        if isinstance(sample_embedding, dict):
            dense_dim = len(sample_embedding['dense_vecs'])
            sparse_dim = len(sample_embedding.get('sparse_vecs', [])) if 'sparse_vecs' in sample_embedding else None
            info = {
                "model_id": os.getenv("MODEL_ID", "BAAI/bge-m3"),
                "dense_dimension": dense_dim,
                "sparse_dimension": sparse_dim,
                "supports_sparse": 'sparse_vecs' in sample_embedding,
                "supports_colbert": 'colbert_vecs' in sample_embedding
            }
        else:
            info = {
                "model_id": os.getenv("MODEL_ID", "BAAI/bge-m3"),
                "dense_dimension": len(sample_embedding),
                "supports_sparse": False,
                "supports_colbert": False
            }
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )

