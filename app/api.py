import os
import sys
import logging
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clustering_api")

# Add project root to sys.path to enable importing from utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from utils.data_preprocessing import clean_text
    from utils.feature_extraction import transform_text
except ImportError as e:
    logger.error(f"Failed to import local utils: {e}")
    raise

# Define Cluster metadata mappings
WIKI_CLUSTERS = {
    0: {
        "topic": "Music, Bands & Singers",
        "keywords": ["album", "band", "music", "record", "release", "song", "play", "single", "singer", "musician"]
    },
    1: {
        "topic": "Team Sports (Football, Baseball, etc.)",
        "keywords": ["season", "league", "play", "game", "football", "team", "coach", "club", "player", "baseball"]
    },
    2: {
        "topic": "Academia, Science & Research",
        "keywords": ["university", "research", "professor", "science", "study", "institute", "book", "phd", "work", "award"]
    },
    3: {
        "topic": "Literature, Writing & Publishing",
        "keywords": ["art", "book", "work", "publish", "novel", "new", "magazine", "write", "artist", "writer"]
    },
    4: {
        "topic": "Film, TV & Acting",
        "keywords": ["film", "music", "television", "award", "work", "theatre", "series", "role", "include", "actor"]
    },
    5: {
        "topic": "Politics, Law & Government",
        "keywords": ["serve", "party", "election", "member", "president", "minister", "law", "elect", "government", "state"]
    },
    6: {
        "topic": "Athletics, Racing & Olympics",
        "keywords": ["championship", "win", "world", "race", "olympic", "team", "finish", "compete", "medal", "event"]
    }
}

NEWS_CLUSTERS = {
    0: {
        "topic": "Hockey & Team Sports",
        "keywords": ["game", "team", "play", "hockey", "player", "year", "win", "good", "playoff", "think"]
    },
    1: {
        "topic": "Middle East Politics & Religion",
        "keywords": ["israel", "people", "armenian", "arab", "jews", "israeli", "say", "think", "right", "know"]
    },
    2: {
        "topic": "PC Hardware & Storage (SCSI/Disk)",
        "keywords": ["drive", "card", "thank", "scsi", "use", "problem", "controller", "disk", "work", "monitor"]
    }
}

app = FastAPI(
    title="Document Clustering API",
    description="FastAPI service for predicting clusters of text using KMeans models trained on Wikipedia Biography and 20 Newsgroups datasets.",
    version="1.0.0"
)

# Paths to models and vectorizers
WIKI_MODEL_PATH = os.path.join(project_root, "models", "kmeans_wiki_model.pkl")
WIKI_VEC_PATH = os.path.join(project_root, "models", "tfidf_wiki_vectorizer.pkl")
NEWS_MODEL_PATH = os.path.join(project_root, "models", "kmeans_news_model.pkl")
NEWS_VEC_PATH = os.path.join(project_root, "models", "tfidf_news_vectorizer.pkl")

# Global dict to store loaded models
models_cache = {}

def load_models():
    """Load models into cache if not already loaded."""
    if "wiki" not in models_cache:
        try:
            logger.info("Loading Wikipedia clustering models...")
            models_cache["wiki_model"] = joblib.load(WIKI_MODEL_PATH)
            # The vectorizer path is used directly in transform_text, but let's make sure it exists
            if not os.path.exists(WIKI_VEC_PATH):
                raise FileNotFoundError(f"Wikipedia TF-IDF vectorizer not found at {WIKI_VEC_PATH}")
            models_cache["wiki"] = True
        except Exception as e:
            logger.error(f"Error loading Wikipedia model: {e}")
            models_cache["wiki"] = False

    if "news" not in models_cache:
        try:
            logger.info("Loading 20 Newsgroups clustering models...")
            models_cache["news_model"] = joblib.load(NEWS_MODEL_PATH)
            if not os.path.exists(NEWS_VEC_PATH):
                raise FileNotFoundError(f"20 Newsgroups TF-IDF vectorizer not found at {NEWS_VEC_PATH}")
            models_cache["news"] = True
        except Exception as e:
            logger.error(f"Error loading News model: {e}")
            models_cache["news"] = False

# Load models at startup
@app.on_event("startup")
def startup_event():
    load_models()

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    text: str = Field(..., description="The raw input text to cluster.", example="He was an American actor who starred in many films.")
    model_type: str = Field(..., description="The model to use: 'wiki' or 'news'.", example="wiki")

class PredictionResponse(BaseModel):
    success: bool
    model_type: str
    original_text: str
    preprocessed_text: str
    cluster_id: int
    topic: str
    keywords: List[str]

@app.get("/")
def read_root():
    """Health check and model status endpoint."""
    load_models()
    return {
        "status": "online",
        "models_loaded": {
            "wiki": models_cache.get("wiki", False),
            "news": models_cache.get("news", False)
        },
        "wiki_clusters_count": len(WIKI_CLUSTERS),
        "news_clusters_count": len(NEWS_CLUSTERS)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predicts the cluster for the provided text."""
    load_models()
    model_type = request.model_type.lower()
    
    if model_type not in ["wiki", "news"]:
        raise HTTPException(status_code=400, detail="Invalid model_type. Must be 'wiki' or 'news'.")
        
    if not models_cache.get(model_type, False):
        raise HTTPException(status_code=503, detail=f"Model for {model_type} is not available/loaded.")
        
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        # 1. Clean the text using the preprocessing utility
        cleaned = clean_text(request.text)
        
        # If preprocessing resulted in empty text, fall back to original text or warn
        if not cleaned.strip():
            cleaned = request.text.lower()
            logger.warning("Preprocessing resulted in empty text, falling back to lowercased raw text.")

        # 2. Vectorize the text using the correct vectorizer path
        vec_path = WIKI_VEC_PATH if model_type == "wiki" else NEWS_VEC_PATH
        X = transform_text(cleaned, model_path=vec_path)
        
        # 3. Predict the cluster using the loaded KMeans model
        model = models_cache[f"{model_type}_model"]
        # predict returns a numpy array
        cluster_id = int(model.predict(X)[0])
        
        # 4. Map cluster ID to metadata
        mapping = WIKI_CLUSTERS if model_type == "wiki" else NEWS_CLUSTERS
        metadata = mapping.get(cluster_id, {"topic": "Unknown", "keywords": []})
        
        return PredictionResponse(
            success=True,
            model_type=model_type,
            original_text=request.text,
            preprocessed_text=cleaned,
            cluster_id=cluster_id,
            topic=metadata["topic"],
            keywords=metadata["keywords"]
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
