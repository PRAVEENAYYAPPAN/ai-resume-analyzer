"""
ResumeAI - Professional Groq Edition
Backend: Groq (Llama 3.3) | Search: Google Cloud Embeddings
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
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc"}

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from environment variables.")
    return Groq(api_key=api_key)

def configure_gemini_embeddings():
    # We still use Gemini for lightweight cloud embeddings (prevents 5GB bundle error)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for Cloud Embeddings component.")
    genai.configure(api_key=api_key)

# ---------------------------------------------------------------------------
# Cloud Embeddings & Similarity (RAG Lite)
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str]) -> list[list[float]]:
    configure_gemini_embeddings()
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return [[0.0] * 768 for _ in texts]

def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2))

# ---------------------------------------------------------------------------
# Utility — Document Parsing
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text: text_parts.append(page_text)
        return "\n".join(text_parts)
    elif ext in ("docx", "doc"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip(): chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ---------------------------------------------------------------------------
# ATS Scoring (Keywords + Semantics)
# ---------------------------------------------------------------------------

TECH_KEYWORDS = ["python", "java", "javascript", "typescript", "react", "node", "aws", "docker", "kubernetes", "sql", "machine learning", "mlops", "llm", "rag"] # Truncated for example

def calculate_ats_score(resume_text, jd_text, resume_emb, jd_emb):
    # Simplified keyword check
    matched = {kw for kw in TECH_KEYWORDS if kw in resume_text.lower() and kw in jd_text.lower()}
    missing = {kw for kw in TECH_KEYWORDS if kw in jd_text.lower() and kw not in resume_text.lower()}
    
    keyword_score = (len(matched) / max(len(matched) + len(missing), 1)) * 100
    semantic_score = cosine_similarity(resume_emb, jd_emb) * 100
    
    ats_score = min(round(0.60 * semantic_score + 0.40 * keyword_score, 1), 98)
    return {
        "ats_score": ats_score,
        "matched_keywords": sorted(list(matched)),
        "missing_keywords": sorted(list(missing)),
        "semantic_score": round(semantic_score, 1),
        "keyword_score": round(keyword_score, 1)
    }

# ---------------------------------------------------------------------------
# Groq LLM Logic
# ---------------------------------------------------------------------------

def analyze_with_groq(prompt: str, max_retries: int = 3) -> dict:
    client = get_groq_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional Career Coach. Return ONLY valid JSON."},
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
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/health")
def health(): return jsonify({"status": "live", "engine": "Groq Llama 3.3", "embeddings": "Google v4"})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files["resume"]
        jd = request.form.get("job_description", "").strip()
        
        # 1. Parse & Chunk
        resume_text = extract_resume_text(file.read(), secure_filename(file.filename))
        chunks = chunk_text(resume_text)
        
        # 2. Embed
        all_embs = get_embeddings([resume_text, jd] + chunks)
        resume_emb, jd_emb, chunk_embs = all_embs[0], all_embs[1], all_embs[2:]
        
        # 3. Retrieve
        sims = [cosine_similarity(jd_emb, c_emb) for c_emb in chunk_embs]
        top_chunks = [chunks[i] for i in np.argsort(sims)[-5:][::-1]]
        
        # 4. Score & Analyze
        ats_data = calculate_ats_score(resume_text, jd, resume_emb, jd_emb)
        prompt = f"Analyze this resume based on these sections: {' '.join(top_chunks)}. JD: {jd[:1000]}. ATS: {ats_data['ats_score']}"
        llm_result = analyze_with_groq(prompt)
        
        return jsonify({**ats_data, **llm_result, "success": True})

    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
