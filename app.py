"""
AI Resume Analyzer - Cloud RAG Edition
Optimized for Vercel / Serverless / Render (Fast & Lightweight)
Using Google Gemini 2.5 Flash + Cloud Embeddings (v004)
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

def configure_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment variables.")
    genai.configure(api_key=api_key)

# ---------------------------------------------------------------------------
# Cloud Embeddings & Similarity (RAG Lite)
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Fetch embeddings from Google Cloud API."""
    configure_genai()
    try:
        # Using the state-of-the-art text-embedding-004 model
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        # Fallback to zeros (prevents crash)
        return [[0.0] * 768 for _ in texts]

def cosine_similarity(v1, v2):
    """Simple dot product similarity for normalized vectors."""
    return float(np.dot(v1, v2))

def semantic_search(query_text: str, chunks: list[str], chunk_embeddings: list[list[float]], top_k: int = 5) -> list[str]:
    """Search chunks using dot product similarity."""
    configure_genai()
    q_emb = genai.embed_content(
        model="models/text-embedding-004",
        content=query_text,
        task_type="retrieval_query"
    )['embedding']
    
    # Calculate similarities
    similarities = [cosine_similarity(q_emb, c_emb) for c_emb in chunk_embeddings]
    
    # Sort and return top_k
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# ---------------------------------------------------------------------------
# Utility — Document Parsing
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    raise ValueError(f"Unsupported file type: {ext}")

# ---------------------------------------------------------------------------
# RAG — Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ---------------------------------------------------------------------------
# ATS Scoring (Keywords + Semantics)
# ---------------------------------------------------------------------------

TECH_KEYWORDS = [
    "python", "java", "javascript", "typescript", "react", "angular", "vue",
    "node", "django", "flask", "fastapi", "sql", "nosql", "mongodb", "postgresql",
    "mysql", "redis", "docker", "kubernetes", "aws", "azure", "gcp", "ci/cd",
    "git", "rest", "graphql", "machine learning", "deep learning", "nlp",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "spark",
    "hadoop", "kafka", "airflow", "data engineering", "mlops", "llm",
    "transformers", "embeddings", "rag", "langchain", "vector database",
    "a/b testing", "agile", "scrum", "linux", "bash", "api", "microservices",
    "html", "css", "tailwind", "figma", "c++", "golang", "rust", "scala",
    "excel", "tableau", "power bi", "statistics", "data analysis",
]

def extract_keywords(text: str) -> set[str]:
    text_lower = text.lower()
    found = set()
    for kw in TECH_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            found.add(kw)
    return found

def calculate_ats_score(resume_text: str, job_description: str, resume_emb: list[float], jd_emb: list[float]) -> dict:
    resume_kws = extract_keywords(resume_text)
    jd_kws = extract_keywords(job_description)
    matched = resume_kws & jd_kws
    missing = jd_kws - resume_kws
    
    # Calculate scores
    keyword_score = (len(matched) / max(len(jd_kws), 1)) * 100 if jd_kws else 50.0
    semantic_score = cosine_similarity(resume_emb, jd_emb) * 100
    semantic_score = min(max(semantic_score, 0), 100)
    
    ats_score = round(0.60 * semantic_score + 0.40 * keyword_score, 1)
    ats_score = min(ats_score, 98)
    
    return {
        "ats_score": ats_score,
        "keyword_score": round(keyword_score, 1),
        "semantic_score": round(semantic_score, 1),
        "matched_keywords": sorted(matched),
        "missing_keywords": sorted(missing),
        "total_jd_keywords": len(jd_kws),
    }

# ---------------------------------------------------------------------------
# Gemini Logic
# ---------------------------------------------------------------------------

def analyze_with_ai(prompt: str, max_retries: int = 3) -> dict:
    configure_genai()
    model = genai.GenerativeModel('gemini-2.5-flash')
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.3
                )
            )
            return json.loads(response.text.strip())
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise e

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "3.0.0", "model": "gemini-2.5-flash-embedded"})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded."}), 400
    file = request.files["resume"]
    job_description = request.form.get("job_description", "").strip()
    if not file or not file.filename:
        return jsonify({"error": "Invalid file."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported type."}), 400
    if len(job_description) < 50:
        return jsonify({"error": "JD too short."}), 400

    try:
        configure_genai()
        # 1. Parse
        file_bytes = file.read()
        filename = secure_filename(file.filename)
        resume_text = extract_resume_text(file_bytes, filename)
        if len(resume_text.strip()) < 100:
            return jsonify({"error": "Could not extract text."}), 400

        # 2. Chunk
        chunks = chunk_text(resume_text)
        
        # 3. Cloud Embeddings (Full Resume, JD, and Chunks)
        # This replaces local SentenceTransformers and FAISS
        all_embeddings = get_embeddings([resume_text, job_description] + chunks)
        resume_emb = all_embeddings[0]
        jd_emb = all_embeddings[1]
        chunk_embeddings = all_embeddings[2:]
        
        # 4. Search & Score
        relevant_chunks = semantic_search(job_description, chunks, chunk_embeddings, top_k=5)
        ats_data = calculate_ats_score(resume_text, job_description, resume_emb, jd_emb)
        
        # 5. Build Prompt (using retrieved RAG context)
        chunks_context = "\n---\n".join(relevant_chunks)
        prompt = f"""You are an expert AI Career Coach and ATS specialist.
Analyze the resume sections against the job description.

## Relevant Resume Sections:
{chunks_context}

## Job Description:
{job_description[:1500]}

## Pre-computed ATS Data:
- ATS Score: {ats_data['ats_score']}/100
- Matched Keywords: {', '.join(ats_data['matched_keywords'][:15]) or 'None'}
- Missing Keywords: {', '.join(ats_data['missing_keywords'][:15]) or 'None'}

Provide a professional analysis in VALID JSON format ONLY. 
{{
  "candidate_summary": "2-3 sentences",
  "overall_verdict": "STRONG_MATCH | GOOD_MATCH | PARTIAL_MATCH | WEAK_MATCH",
  "verdict_explanation": "1-2 sentences",
  "strengths": [ {{"title": "Title", "description": "Desc"}} ],
  "improvement_areas": [ {{"title": "Title", "description": "Desc", "priority": "HIGH|MEDIUM|LOW"}} ],
  "missing_skills": ["skill1"],
  "resume_score_breakdown": {{ "experience_relevance": 0, "skills_alignment": 0, "education_fit": 0, "achievements_impact": 0 }},
  "quick_wins": ["tip1"],
  "interview_talking_points": ["pt1"],
  "rewrite_suggestion": "example rewrite"
}}
Return ONLY JSON."""

        llm_result = analyze_with_ai(prompt)

        return jsonify({
            **ats_data,
            **llm_result,
            "success": True
        }), 200

    except Exception as e:
        logger.exception(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
