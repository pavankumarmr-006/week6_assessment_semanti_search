
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# CONFIGURATION
# ==========================================

API_KEY = "my-secret-key"
MODEL_NAME = "all-MiniLM-L6-v2"

model = None
document_embeddings = None


# ==========================================
# DOCUMENTS
# ==========================================

documents = [
    {
        "text": "Employees are entitled to 20 days of annual leave per year.",
        "doc_type": "HR"
    },
    {
        "text": "Employees can apply for sick leave when they are ill.",
        "doc_type": "HR"
    },
    {
        "text": "Annual leave applications must be approved by the manager.",
        "doc_type": "HR"
    },
    {
        "text": "Employees receive maternity leave according to company policy.",
        "doc_type": "HR"
    },
    {
        "text": "Employees must submit leave requests through the HR portal.",
        "doc_type": "HR"
    }
]


# ==========================================
# MODEL INITIALIZATION
# ==========================================

def initialize_model():

    global model, document_embeddings

    if model is None:

        print("Loading Sentence Transformer model...")

        model = SentenceTransformer(MODEL_NAME)

        texts = [doc["text"] for doc in documents]

        document_embeddings = model.encode(
            texts,
            convert_to_numpy=True
        )

        print("Model loaded successfully!")


# ==========================================
# FASTAPI LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    initialize_model()

    yield


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Semantic Search API",
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# REQUEST MODEL
# ==========================================


# REQUEST MODEL

class SearchRequest(BaseModel):

    query: str = Field(..., min_length=1)

    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )

    doc_type: str | None = None

    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )


# HEALTH CHECK

@app.get("/health")
def health():

    return {
        "status": "running"
    }


@app.get("/health/message")
def health_message():

    return {
        "message": "Semantic Search API is running successfully"
    }



# ==========================================
# API KEY VALIDATION
# ==========================================

def check_api_key(api_key):

    if api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "running"
    }


@app.get("/health/message")
def health_message():

    return {
        "message": "Semantic Search API is running successfully"
    }


# ==========================================
# SEARCH ENDPOINT
# ==========================================

@app.post("/search")
def search(
    request: SearchRequest,
    x_api_key: str = Header(None)
):

    # Check API key
    check_api_key(x_api_key)

    # Initialize model if lifespan has not run
    initialize_model()

    # Validate query
    if not request.query.strip():

        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty"
        )

    # Generate query embedding
    query_embedding = model.encode(
        request.query,
        convert_to_numpy=True
    )

    # Calculate cosine similarity
    scores = cosine_similarity(
        [query_embedding],
        document_embeddings
    )[0]

    results = []

    for index, score in enumerate(scores):

        doc = documents[index]

        # Filter document type
        if (
            request.doc_type is not None
            and doc["doc_type"] != request.doc_type
        ):
            continue

        # Filter minimum score
        if score < request.min_score:
            continue

        results.append({
            "text": doc["text"],
            "doc_type": doc["doc_type"],
            "score": float(score)
        })

    # Sort results
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Return top K results
    results = results[:request.top_k]

    return {
        "query": request.query,
        "top_k": request.top_k,
        "results": results
    }
