"""
AI Resume Analyzer — Pure Groq Edition
LLM: Groq Llama 3.3 | Search: Local Llama-Lite Embeddings
Deployment: Render (Backend) / Vercel (Frontend)
"""

import os
import re
import json
import io
import logging
import time
from pathlib import Path
from functools import lru_cache

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import pdfplumber
from docx import Document
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration (Pure Groq)
# ---------------------------------------------------------------------------
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Professional CORS to allow Vercel
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc"}

# Global model cache (Lazy loading)
_EMBED_MODEL = None

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from environment variables.")
    return Groq(api_key=api_key)

def get_embed_model():
    """Lazily load the embedding model to save startup time."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        logger.info("Initializing 'all-MiniLM-L6-v2' (Lite Local model)...")
        # We use CPU-only mode for Render compat
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL

# ---------------------------------------------------------------------------
# Semantic Search (Lite Llama RAG)
# ---------------------------------------------------------------------------

def calculate_embeddings(texts: list[str]) -> np.ndarray:
    model = get_embed_model()
    return model.encode(texts, normalize_embeddings=True)

def semantic_search(query: str, chunks: list[str], chunk_embeddings: np.ndarray, top_k: int = 5) -> list[str]:
    model = get_embed_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    
    # Calculate cosine similarities using dot product (since normalized)
    similarities = np.dot(chunk_embeddings, q_emb)
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text: text_parts.append(text)
        return "\n".join(text_parts)
    elif ext in ("docx", "doc"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# ---------------------------------------------------------------------------
# Groq Logic (LLM Analysis)
# ---------------------------------------------------------------------------

def analyze_with_groq(prompt: str, max_retries: int = 3) -> dict:
    client = get_groq_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional ATS expert. JSON ONLY."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise e

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "live", "engine": "Groq Llama 3.3", "embeddings": "Local-Lite"})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "Resume missing"}), 400
    file = request.files["resume"]
    jd = request.form.get("job_description", "").strip()
    
    try:
        # 1. Parse
        resume_text = extract_resume_text(file.read(), secure_filename(file.filename))
        chunks = chunk_text(resume_text)
        
        # 2. Embed & Retrieve
        chunk_embs = calculate_embeddings(chunks)
        relevant_chunks = semantic_search(jd, chunks, chunk_embs)
        
        # 3. Analyze with Groq
        context_str = "\n---\n".join(relevant_chunks)
        prompt = f"Analyze resume chunks for JD. JD: {jd[:1000]}. Chunks: {context_str}"
        llm_result = analyze_with_groq(prompt)
        
        # 4. Mix in ATS scoring (simple placeholder for now)
        ats_score = min(round(np.mean([abs(chunk_embs[0][0]*100), 75]), 1), 98) # Mock score logic
        
        return jsonify({
            **llm_result,
            "ats_score": ats_score,
            "success": True
        }), 200

    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
