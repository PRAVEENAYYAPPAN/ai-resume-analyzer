"""
AI Resume Analyzer — Pure Groq Edition
Engine: Indestructible "Zero-Library" RAG (Numpy only)
Build Speed: Lightning | Stability: Infinite
"""

import os
import re
import json
import io
import logging
import time
from pathlib import Path
from math import sqrt, log

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import pdfplumber
from docx import Document
import numpy as np
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

STOPWORDS = {"the", "and", "for", "with", "from", "that", "this", "which", "are", "have", "you", "not", "but"}

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from environment variables.")
    return Groq(api_key=api_key)

# ---------------------------------------------------------------------------
# Professional "Zero-Library" RAG Implementation (TF Similarity)
# ---------------------------------------------------------------------------

def get_words(text: str) -> list[str]:
    """Clean and tokenize text."""
    return [w for w in re.sub(r'[^a-zA-Z\s]', '', text.lower()).split() if w and w not in STOPWORDS]

def calculate_similarity_lite(query: str, chunk: str) -> float:
    """Calculate professional-grade similarity WITHOUT heavy libraries."""
    q_words = get_words(query)
    c_words = get_words(chunk)
    
    if not q_words or not c_words: return 0.0
    
    # Simple Term Frequency (TF)
    all_words = list(set(q_words + c_words))
    vec_q = [q_words.count(w) for w in all_words]
    vec_c = [c_words.count(w) for w in all_words]
    
    # Cosine Similarity
    dot = sum(a*b for a, b in zip(vec_q, vec_c))
    norm_q = sqrt(sum(a*a for a in vec_q))
    norm_c = sqrt(sum(b*b for b in vec_c))
    
    if norm_q == 0 or norm_c == 0: return 0.0
    return dot / (norm_q * norm_c)

def semantic_search_lite(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    """Fast, library-free RAG ranking."""
    if not chunks: return []
    scored = [(calculate_similarity_lite(query, c), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "pdf":
            text_parts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text: text_parts.append(text)
            return "\n".join(text_parts).strip()
        elif ext in ("docx", "doc"):
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    except Exception as e:
        logger.error(f"Extraction error: {e}")
    return ""

def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size - 40)]

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
                time.sleep(2)
                continue
            raise e

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "live", "engine": "Groq Llama 3.3", "rag": "Indestructible-Lite"})

@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS": return jsonify({}), 200
    if "resume" not in request.files:
        return jsonify({"success": False, "error": "Resume missing"}), 400
    
    file = request.files["resume"]
    jd = request.form.get("job_description", "").strip() or "General Resume Analysis"
    
    try:
        # 1. Parse
        resume_text = extract_resume_text(file.read(), secure_filename(file.filename))
        if not resume_text or len(resume_text) < 10:
            return jsonify({"success": False, "error": "Could not extract sufficient text from resume."}), 400
            
        # 2. RAG Search (Indestructible version)
        chunks = chunk_text(resume_text)
        relevant_chunks = semantic_search_lite(jd, chunks)
        
        # 3. Analyze with Groq
        context_str = "\n---\n".join(relevant_chunks)
        prompt = f"Analyze resume for JD. Return valid JSON. JD: {jd[:1000]}. Context from resume: {context_str}"
        analysis = analyze_with_groq(prompt)
        
        return jsonify({**analysis, "success": True}), 200

    except Exception as e:
        logger.exception(e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
