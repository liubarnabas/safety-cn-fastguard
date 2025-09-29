# server/prefilter_service.py
# -*- coding: utf-8 -*-
"""
Safety Prefilter (Binary) FastAPI Service

A high-performance binary safety classification service that provides:
- Batch prediction endpoint: /score
- Health check endpoint: /healthz
- Automatic loading of temperature.json (T) and theta.json (threshold)
- Priority ONNX Runtime (if model.onnx exists), fallback to PyTorch
- Compatible startup from any working directory (auto-adds project root to sys.path)

Environment Variables:
    PREFILTER_PATH: Model directory path (default: ./artifacts/prefilter_roberta)
    USE_ONNX: Use ONNX runtime if available (default: 1/true)
    HOST: Service host (default: 0.0.0.0)
    PORT: Service port (default: 8080)
    MAX_BATCH_SIZE: Maximum batch size for requests (default: 100)
    LOG_LEVEL: Logging level (default: INFO)
"""

import os
import sys
import time
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- Add project root to sys.path ----------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # server/ parent dir
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
    logger.info(f"Added project root to sys.path: {_PROJECT_ROOT}")

# ---------- Safe import with error handling ----------
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    logger.info("FastAPI dependencies imported successfully")
except ImportError as e:
    logger.error(f"Failed to import FastAPI dependencies: {e}")
    raise

try:
    from src.models.prefilter_binary import Prefilter
    logger.info("Prefilter model imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Prefilter model: {e}")
    logger.warning("Continuing without model - service will be in degraded mode")
    Prefilter = None

APP_TITLE = "Safety Prefilter (Binary)"
APP_DESC = "High-performance SAFE/UNSAFE binary classification prefilter service with batch processing, temperature scaling, and ONNX/CPU/GPU support."
APP_VER = "1.1.0"

# Configuration constants
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))

class ScoreRequest(BaseModel):
    """Request model for batch text classification."""
    texts: List[str] = Field(..., min_items=1, max_items=MAX_BATCH_SIZE,
                             description=f"List of texts to classify (max {MAX_BATCH_SIZE} items)")
    temperature: Optional[float] = Field(None, ge=0.1, le=10.0,
                                       description="Optional temperature for softmax scaling (0.1-10.0)")

    @validator('texts')
    def validate_texts(cls, v):
        """Validate text inputs."""
        if not v:
            raise ValueError("texts list cannot be empty")

        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Item {i} must be a string")
            if len(text.strip()) == 0:
                raise ValueError(f"Item {i} cannot be empty or whitespace only")
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"Item {i} exceeds maximum length of {MAX_TEXT_LENGTH} characters")

        return [text.strip() for text in v]

class ScoreResponse(BaseModel):
    """Response model for batch text classification."""
    probs: List[float] = Field(..., description="UNSAFE probabilities for each input text")
    labels: List[int] = Field(..., description="Binary labels: 1=UNSAFE, 0=SAFE")
    theta: float = Field(..., description="Classification threshold used")
    T: float = Field(..., description="Temperature scaling factor used")
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded successfully")
    theta: Optional[float] = Field(None, description="Current threshold value")
    T: Optional[float] = Field(None, description="Current temperature value")
    model_path: str = Field(..., description="Model directory path")
    use_onnx: bool = Field(..., description="Whether ONNX runtime is being used")
    uptime: float = Field(..., description="Service uptime in seconds")

# OpenAI-compatible models
class ChatMessage(BaseModel):
    """OpenAI-compatible message model."""
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(..., description="Model name")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(None, ge=0.1, le=10.0, description="Temperature for response generation")

class ChatChoice(BaseModel):
    """OpenAI-compatible choice model."""
    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Response message")
    finish_reason: str = Field(..., description="Reason for completion finish")

class ChatUsage(BaseModel):
    """OpenAI-compatible usage model."""
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")
    total_tokens: int = Field(..., description="Total number of tokens")

class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(..., description="Unique response ID")
    object: str = Field(..., description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[ChatChoice] = Field(..., description="Response choices")
    usage: ChatUsage = Field(..., description="Token usage information")

# FastAPI app initialization
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESC,
    version=APP_VER,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Environment configuration
PREFILTER_PATH = os.getenv("PREFILTER_PATH", "./artifacts/prefilter_roberta")
USE_ONNX = str(os.getenv("USE_ONNX", "1")).lower() in {"1", "true", "yes"}

# Global variables for service state
pref = None
service_start_time = time.time()

# Initialize model with error handling
def initialize_model():
    """Initialize the prefilter model with proper error handling."""
    global pref

    if Prefilter is None:
        logger.error("Prefilter class not available - cannot initialize model")
        return False

    try:
        logger.info(f"Initializing model from path: {PREFILTER_PATH}")
        logger.info(f"ONNX mode: {USE_ONNX}")

        pref = Prefilter(path=PREFILTER_PATH, use_onnx=USE_ONNX)
        logger.info("Model initialized successfully")
        logger.info(f"Model theta: {getattr(pref, 'theta', 'N/A')}")
        logger.info(f"Model temperature: {getattr(pref, 'T', 'N/A')}")
        return True

    except FileNotFoundError as e:
        logger.error(f"Model files not found at {PREFILTER_PATH}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False

# Initialize model on startup
model_loaded = initialize_model()

@app.get("/healthz", response_model=HealthResponse)
def healthz():
    """Comprehensive health check endpoint."""
    uptime = time.time() - service_start_time

    response_data = {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": PREFILTER_PATH,
        "use_onnx": USE_ONNX,
        "uptime": uptime
    }

    if pref is not None:
        try:
            response_data["theta"] = float(pref.theta)
            response_data["T"] = float(pref.T)
        except Exception as e:
            logger.warning(f"Error accessing model parameters in health check: {e}")
            response_data["theta"] = None
            response_data["T"] = None
    else:
        response_data["theta"] = None
        response_data["T"] = None

    return HealthResponse(**response_data)

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """
    Classify batch of texts as SAFE/UNSAFE.

    Args:
        req: ScoreRequest containing texts and optional temperature

    Returns:
        ScoreResponse with probabilities, labels, and metadata

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    start_time = time.time()

    # Check if model is available
    if not model_loaded or pref is None:
        logger.error("Model not loaded - cannot process prediction request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Service is in degraded mode."
        )

    try:
        # Log request info
        logger.info(f"Processing batch of {len(req.texts)} texts")
        if req.temperature is not None:
            logger.debug(f"Using custom temperature: {req.temperature}")

        # Perform prediction
        probs = pref.predict_proba(req.texts, T=req.temperature)

        # Calculate labels using threshold
        theta = float(pref.theta)
        labels = [1 if p >= theta else 0 for p in probs]

        # Determine effective temperature
        T_eff = float(pref.T if req.temperature is None else req.temperature)

        processing_time = time.time() - start_time

        # Log results
        unsafe_count = sum(labels)
        logger.info(f"Batch processed: {unsafe_count}/{len(labels)} classified as UNSAFE, "
                   f"processing time: {processing_time:.3f}s")

        return ScoreResponse(
            probs=probs,
            labels=labels,
            theta=theta,
            T=T_eff,
            processing_time=processing_time
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error during prediction: {e}, processing time: {processing_time:.3f}s")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint for safety classification.

    This endpoint processes chat messages and returns safety classification results
    in a format compatible with OpenAI's API structure.

    Args:
        req: ChatCompletionRequest with model, messages, and optional temperature

    Returns:
        ChatCompletionResponse with safety classification results

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    import uuid
    start_time = time.time()

    # Check if model is available
    if not model_loaded or pref is None:
        logger.error("Model not loaded - cannot process chat completion request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Service is in degraded mode."
        )

    try:
        # Extract text content from messages (only user messages)
        texts = []
        for message in req.messages:
            if message.role == "user":
                texts.append(message.content)

        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user messages found in request"
            )

        logger.info(f"Processing chat completion with {len(texts)} user messages")

        # Perform prediction on all user message texts
        probs = pref.predict_proba(texts, T=req.temperature)

        # Calculate labels using threshold
        theta = float(pref.theta)
        labels = [1 if p >= theta else 0 for p in probs]

        # Determine effective temperature
        T_eff = float(pref.T if req.temperature is None else req.temperature)

        # Create response content based on safety classification
        unsafe_count = sum(labels)
        if unsafe_count > 0:
            safety_status = "UNSAFE"
            response_content = f"Content safety check detected {unsafe_count} unsafe message(s). Probability scores: {probs}"
        else:
            safety_status = "SAFE"
            response_content = f"Content safety check passed. All {len(texts)} messages are safe. Probability scores: {probs}"

        # Calculate token usage (rough estimation)
        total_input_chars = sum(len(text) for text in texts)
        prompt_tokens = total_input_chars // 4  # Rough estimation: 4 chars per token
        completion_tokens = len(response_content) // 4
        total_tokens = prompt_tokens + completion_tokens

        processing_time = time.time() - start_time

        # Create OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            object="chat.completion",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_content
                    ),
                    finish_reason="stop"
                )
            ],
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )

        logger.info(f"Chat completion processed: {safety_status}, "
                   f"processing time: {processing_time:.3f}s")

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error during chat completion: {e}, processing time: {processing_time:.3f}s")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    import socket

    logger.info(f"Starting {APP_TITLE} v{APP_VER}")
    logger.info(f"Model path: {PREFILTER_PATH}")
    logger.info(f"ONNX mode: {USE_ONNX}")
    logger.info(f"Max batch size: {MAX_BATCH_SIZE}")
    logger.info(f"Max text length: {MAX_TEXT_LENGTH}")
    logger.info(f"Model loaded: {model_loaded}")
    logger.info("Available endpoints: /score, /healthz, /v1/chat/completions")

    # Log network information
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(f"Host information:")
        logger.info(f"  - Hostname: {hostname}")
        logger.info(f"  - Local IP: {local_ip}")
        logger.info(f"  - Binding to: 0.0.0.0:8080 (all interfaces)")
        logger.info(f"  - External access: http://{local_ip}:8080")
    except Exception as e:
        logger.warning(f"Could not determine network info: {e}")

    logger.info("Server is ready to accept connections from any IP address")

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    uptime = time.time() - service_start_time
    logger.info(f"Shutting down {APP_TITLE} after {uptime:.2f} seconds")

if __name__ == "__main__":
    import uvicorn

    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Workers: {workers}, Reload: {reload}")
    logger.info(f"Service will be accessible at:")
    logger.info(f"  - Local: http://127.0.0.1:{port}")
    logger.info(f"  - Network: http://{host}:{port}")
    logger.info(f"  - All interfaces: http://0.0.0.0:{port}")

    try:
        # Single process is recommended for ML models to avoid memory overhead
        # For multi-worker deployment, use gunicorn/uvicorn at container/PM2 level
        uvicorn.run(
            "prefilter_service:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
            access_log=True,
            server_header=False,
            date_header=False
        )
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is already in use. Please stop the existing service or use a different port.")
            logger.info("You can check what's using the port with: netstat -tulpn | grep :8080")
        elif "Permission denied" in str(e):
            logger.error(f"Permission denied binding to port {port}. Try a port > 1024 or run with appropriate permissions.")
        else:
            logger.error(f"Failed to start server: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error starting server: {e}")
        raise
