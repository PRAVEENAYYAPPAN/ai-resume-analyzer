"""
AI Resume Analyzer - Backend
RAG + LLM powered resume analysis using FAISS embeddings + Google Gemini
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
import faiss
from sentence_transformers import SentenceTransformer
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Lazy-load Heavy Models
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer model …")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer loaded.")
    return _embedding_model

def get_gemini_model():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")
    genai.configure(api_key=GEMINI_API_KEY)
    # Using Gemini 2.5 Flash as requested for updated tech
    return genai.GenerativeModel('gemini-2.5-flash')

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
# RAG — Chunk + Embed + FAISS Search
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

def build_faiss_index(chunks: list[str], model: SentenceTransformer):
    embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

def semantic_search(query: str, chunks: list[str], index, model: SentenceTransformer, top_k: int = 5) -> list[str]:
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def compute_rag_similarity(resume_text: str, job_description: str, model: SentenceTransformer) -> float:
    r_emb = model.encode([resume_text], normalize_embeddings=True)
    j_emb = model.encode([job_description], normalize_embeddings=True)
    similarity = float(np.dot(r_emb[0], j_emb[0]))
    return round(min(max(similarity * 100, 0), 100), 2)

# ---------------------------------------------------------------------------
# ATS Scoring
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

def calculate_ats_score(resume_text: str, job_description: str, semantic_score: float) -> dict:
    resume_kws = extract_keywords(resume_text)
    jd_kws = extract_keywords(job_description)
    matched = resume_kws & jd_kws
    missing = jd_kws - resume_kws
    keyword_score = (len(matched) / max(len(jd_kws), 1)) * 100 if jd_kws else 50.0
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
# Gemini Analysis
# ---------------------------------------------------------------------------

def build_analysis_prompt(resume_text: str, job_description: str, ats_data: dict, relevant_chunks: list[str]) -> str:
    chunks_context = "\n---\n".join(relevant_chunks[:5])
    return f"""You are an expert AI Career Coach and ATS specialist.
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
Structure:
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
Return ONLY the JSON object."""

def analyze_with_ai(prompt: str, max_retries: int = 3) -> dict:
    model = get_gemini_model()
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
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "2.5.0", "model": "gemini-2.5-flash"})

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
        file_bytes = file.read()
        filename = secure_filename(file.filename)
        resume_text = extract_resume_text(file_bytes, filename)
        if len(resume_text.strip()) < 100:
            return jsonify({"error": "Could not extract text."}), 400

        model = get_embedding_model()
        chunks = chunk_text(resume_text)
        index, _ = build_faiss_index(chunks, model)
        relevant_chunks = semantic_search(job_description, chunks, index, model, top_k=5)
        semantic_score = compute_rag_similarity(resume_text, job_description, model)
        ats_data = calculate_ats_score(resume_text, job_description, semantic_score)
        
        prompt = build_analysis_prompt(resume_text, job_description, ats_data, relevant_chunks)
        llm_result = analyze_with_ai(prompt)

        return jsonify({
            "success": True,
            "ats_score": ats_data["ats_score"],
            "keyword_score": ats_data["keyword_score"],
            "semantic_score": ats_data["semantic_score"],
            "matched_keywords": ats_data["matched_keywords"],
            "missing_keywords": ats_data["missing_keywords"],
            **llm_result
        }), 200
    except Exception as e:
        logger.exception(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
