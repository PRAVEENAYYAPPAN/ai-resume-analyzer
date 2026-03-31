# 🧠 AI Resume Analyzer — RAG + LLM Powered

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=for-the-badge&logo=flask)
![Gemini](https://img.shields.io/badge/Gemini%201.5%20Flash-LLM-orange?style=for-the-badge&logo=google)
![FAISS](https://img.shields.io/badge/FAISS-RAG%20Vector%20DB-purple?style=for-the-badge)
![Render](https://img.shields.io/badge/Deploy-Render.com-46E3B7?style=for-the-badge&logo=render)

**A production-ready AI Resume Analyzer using Retrieval-Augmented Generation (RAG) and Large Language Models (LLM).**

Upload your resume → Get ATS score, skill gap analysis & personalized improvement roadmap in seconds.

[🚀 Live Demo](#) · [📖 Documentation](#how-it-works) · [⚙️ Deploy Your Own](#deployment)

</div>

---

## ✨ Features

| Feature | Technology |
|---|---|
| 📄 Resume Parsing | `pdfplumber` (PDF), `python-docx` (DOCX) |
| 🧬 Semantic Embeddings | `sentence-transformers` — all-MiniLM-L6-v2 |
| 🔍 RAG Vector Search | `FAISS` — inner-product indexed chunks |
| 🤖 LLM Analysis | Google `Gemini 1.5 Flash` |
| 📊 ATS Score | Keyword match + semantic similarity blend |
| 💡 Smart Suggestions | AI-generated improvement roadmap |
| 🎨 Premium UI | Dark glassmorphism with particle animations |
| ☁️ One-click Deploy | Render.com free tier |

---

## 🏗️ Architecture

```
Resume (PDF/DOCX)  +  Job Description
         │
         ▼
┌─────────────────────────────────────┐
│        Flask REST API (/api/analyze) │
│                                     │
│  1. Parse → pdfplumber / python-docx│
│  2. Chunk text (150-word overlaps)  │
│  3. Embed → SentenceTransformers   │
│  4. FAISS Index → Semantic Search  │   ◄── RAG
│  5. ATS Score (keyword + semantic) │
│  6. Gemini 1.5 Flash analysis      │   ◄── LLM
│                                     │
└───────────────┬─────────────────────┘
                │ JSON Response
                ▼
        Premium Web Frontend
  (ATS gauge · Skill chips · Cards)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API Key ([Get free key](https://aistudio.google.com/))

### Local Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the app
python app.py
```

Visit **http://localhost:5000** in your browser.

---

## ☁️ Deployment (Render.com — Free Tier)

### One-click deployment:

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AI Resume Analyzer"
   git remote add origin https://github.com/yourusername/ai-resume-analyzer.git
   git push -u origin main
   ```

2. **Deploy on Render.com:**
   - Go to [render.com](https://render.com) → New → Web Service
   - Connect your GitHub repository
   - Render auto-detects `render.yaml` — all settings pre-configured
   - Add environment variable: `GEMINI_API_KEY = your_key_here`
   - Click **Deploy**

3. **Your live URL:** `https://ai-resume-analyzer-xxxx.onrender.com`

> **Note:** First cold start may take ~60s as models download. Subsequent requests are fast.

---

## 📡 API Reference

### `POST /api/analyze`

Analyze a resume against a job description.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|---|---|---|---|
| `resume` | File | ✅ | PDF or DOCX resume (max 5MB) |
| `job_description` | String | ✅ | Full job description (min 50 chars) |

**Response:** `application/json`
```json
{
  "success": true,
  "ats_score": 78.5,
  "semantic_score": 82.1,
  "keyword_score": 71.3,
  "matched_keywords": ["python", "machine learning", "flask"],
  "missing_keywords": ["kubernetes", "terraform"],
  "overall_verdict": "GOOD_MATCH",
  "candidate_summary": "...",
  "strengths": [{"title": "...", "description": "..."}],
  "improvement_areas": [{"title": "...", "description": "...", "priority": "HIGH"}],
  "quick_wins": ["...", "..."],
  "interview_talking_points": ["...", "..."],
  "rewrite_suggestion": "..."
}
```

### `GET /api/health`
Returns API status and version info.

---

## 🧪 How It Works

### Stage 1: Parse & Chunk
The resume is parsed from PDF/DOCX format and split into overlapping 150-word chunks to preserve context boundaries.

### Stage 2: Embed & Retrieve (RAG)
- Each chunk is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`
- Vectors are stored in a **FAISS** in-memory index with inner-product similarity
- The job description queries the index to retrieve the top-5 most relevant resume sections ← **This is the RAG step**

### Stage 3: ATS Scoring
- **Keyword Score (40%):** Regex-based detection of 80+ technical keywords from the JD
- **Semantic Score (60%):** Cosine similarity between full resume and JD embeddings
- Blended into a realistic ATS score (capped at 98 to be honest)

### Stage 4: LLM Analysis
The retrieved RAG context + full resume + JD are sent to **Gemini 1.5 Flash** with a structured JSON prompt for:
- Candidate summary & verdict
- Score breakdown (experience, skills, education, achievements)
- Strengths and improvement areas with priority levels
- Quick wins and interview talking points
- AI-powered bullet point rewrite suggestion

---

## 📁 Project Structure

```
ai-resume-analyzer/
├── app.py                  # Flask backend — RAG + LLM pipeline
├── requirements.txt        # Python dependencies
├── Procfile               # Gunicorn startup for Render.com
├── render.yaml            # Render.com deployment config
├── .env.example           # Environment variable template
├── templates/
│   └── index.html         # Premium dark-mode frontend
├── static/
│   ├── style.css          # Glassmorphism dark UI
│   └── app.js             # Particles, drag-drop, Chart.js
└── README.md
```

---

## 🔒 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | ✅ |
| `SECRET_KEY` | Flask session secret | Recommended |
| `FLASK_ENV` | `production` or `development` | Optional |
| `PORT` | Server port (default: 5000) | Optional |

---

## 🛡️ Security Notes

- Resume files are processed in-memory (never saved to disk)
- All uploads are validated for type and size (max 5MB)
- API keys are loaded via environment variables (never committed)
- CORS is enabled for frontend/backend communication

---

## 🤝 Skills Demonstrated

This project demonstrates:
- ✅ **RAG Pipeline** — FAISS + Sentence Transformers
- ✅ **LLM Integration** — Google Gemini API with structured output
- ✅ **Embeddings** — Dense vector representations for semantic search
- ✅ **REST API Design** — Flask with proper error handling
- ✅ **Frontend Engineering** — Vanilla JS + Canvas animations
- ✅ **Cloud Deployment** — Render.com with `render.yaml`
- ✅ **Production Ready** — Gunicorn, env vars, lazy model loading

---

## 📄 License

MIT License — free to use, modify, and deploy.

---

<div align="center">
  Built for Junior AI Developer portfolios · RAG + LLM + Embeddings + Deployment
</div>
