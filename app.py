"""
AI Resume Analyzer — Pure Groq Llama 3.3
Search: Lite Professional TF-IDF RAG (Stability Patch)
No Torch/Memory Crashes on Render Free Tier.
"""

import os
import re
import json
import io
import logging
import time
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import pdfplumber
from docx import Document
import numpy as np
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Total CORS freedom for Vercel connection
CORS(app, resources={r"/*": {"origins": "*"}})

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from environment variables.")
    return Groq(api_key=api_key)

# ---------------------------------------------------------------------------
# Professional Lite RAG (TF-IDF Similarity)
# ---------------------------------------------------------------------------

def semantic_search_lite(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    """Extremely memory-efficient RAG using TF-IDF."""
    if not chunks: return []
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks)
    query_vec = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

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
            return "\n".join(text_parts)
        elif ext in ("docx", "doc"):
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"Text extraction error: {e}")
    return ""

def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size - 30)]

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "live", "engine": "Groq Llama 3.3", "rag": "TF-IDF Lite"})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    # CORS headers for stability
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if "resume" not in request.files:
        return jsonify({"success": False, "error": "Resume missing"}), 400
    
    file = request.files["resume"]
    jd = request.form.get("job_description", "").strip()
    
    try:
        # 1. Parse
        resume_text = extract_resume_text(file.read(), secure_filename(file.filename))
        if not resume_text:
            return jsonify({"success": False, "error": "Could not extract text from file."}), 400
            
        # 2. RAG Search (Lite version)
        chunks = chunk_text(resume_text)
        relevant_chunks = semantic_search_lite(jd, chunks)
        
        # 3. Analyze with Groq
        context_str = "\n---\n".join(relevant_chunks)
        prompt = f"Analyze resume for JD. Return valid JSON. Output schema: ats_score (0-100), overall_verdict (STRONG_MATCH, etc), candidate_summary, verdict_explanation, matched_keywords (list), missing_keywords (list), strengths (list of title/desc), improvement_areas (list of title/desc/priority), quick_wins (list), interview_talking_points (list). JD: {jd[:1000]}. Context from resume: {context_str}"
        
        analysis = analyze_with_groq(prompt)
        
        return jsonify({**analysis, "success": True}), 200

    except Exception as e:
        logger.exception(e)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
