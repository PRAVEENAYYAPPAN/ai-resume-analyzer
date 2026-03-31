/* ============================================================
   AI Resume Analyzer — App Logic
   ============================================================ */

"use strict";

// ── Particle Background ──────────────────────────────────────
(function initParticles() {
  const canvas = document.getElementById("particles-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function createParticles(count = 80) {
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.5 + 0.3,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      a: Math.random() * 0.6 + 0.1,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(139,92,246,${p.a})`;
      ctx.fill();

      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;
    }
    requestAnimationFrame(draw);
  }

  resize();
  createParticles();
  draw();
  window.addEventListener("resize", () => { resize(); createParticles(); });
})();


// ── Loading Step Cycle ───────────────────────────────────────
let stepTimer = null;

function startLoadingSteps() {
  let current = 0;
  const items = document.querySelectorAll(".step-item");
  items.forEach(el => { el.classList.remove("active", "done"); });
  items[0].classList.add("active");

  stepTimer = setInterval(() => {
    if (current < items.length - 1) {
      items[current].classList.remove("active");
      items[current].classList.add("done");
      current++;
      items[current].classList.add("active");
    }
  }, 3500);
}

function stopLoadingSteps() {
  clearInterval(stepTimer);
  document.querySelectorAll(".step-item").forEach(el => {
    el.classList.add("done");
    el.classList.remove("active");
  });
}


// ── Drop Zone Logic ──────────────────────────────────────────
const dropZone     = document.getElementById("drop-zone");
const fileInput    = document.getElementById("resume-file");
const dropContent  = document.getElementById("drop-zone-content");
const fileSelected = document.getElementById("file-selected");
const fileNameEl   = document.getElementById("file-name");
const fileSizeEl   = document.getElementById("file-size");
const fileRemoveBtn= document.getElementById("file-remove");

let selectedFile = null;

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(2) + " MB";
}

function showFile(file) {
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = formatBytes(file.size);
  dropContent.classList.add("hidden");
  fileSelected.classList.remove("hidden");
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  dropContent.classList.remove("hidden");
  fileSelected.classList.add("hidden");
}

dropZone.addEventListener("click", (e) => {
  if (e.target === fileRemoveBtn || fileRemoveBtn.contains(e.target)) return;
  fileInput.click();
});

dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) showFile(fileInput.files[0]);
});

fileRemoveBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearFile();
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (["pdf","docx","doc"].includes(ext)) {
      showFile(file);
    } else {
      showError("Unsupported file type. Please drop a PDF or DOCX file.");
    }
  }
});


// ── Character Counter ────────────────────────────────────────
const textarea  = document.getElementById("job-description");
const charCount = document.getElementById("char-count");

textarea.addEventListener("input", () => {
  const len = textarea.value.length;
  charCount.textContent = len;
  charCount.style.color = len >= 50 ? "var(--clr-green-l)" : "var(--clr-red-l)";
});


// ── Error Toast ──────────────────────────────────────────────
function showError(msg) {
  const toast = document.getElementById("error-toast");
  document.getElementById("error-message").textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 6000);
}

document.getElementById("toast-close").addEventListener("click", () => {
  document.getElementById("error-toast").classList.add("hidden");
});


// ── Section Visibility ───────────────────────────────────────
function show(id)   { document.getElementById(id).classList.remove("hidden"); }
function hide(id)   { document.getElementById(id).classList.add("hidden"); }

function showLoading() {
  hide("results-section");
  show("loading-section");
  startLoadingSteps();
  document.getElementById("loading-section").scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideLoading() {
  stopLoadingSteps();
  hide("loading-section");
}


// ── Gauge Chart ──────────────────────────────────────────────
let gaugeChart = null;

function renderGauge(score) {
  const ctx = document.getElementById("gauge-chart").getContext("2d");
  if (gaugeChart) gaugeChart.destroy();

  const color = score >= 75 ? "#10b981"
              : score >= 55 ? "#3b82f6"
              : score >= 35 ? "#eab308"
              : "#ef4444";

  gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [score, 100 - score],
        backgroundColor: [color, "rgba(255,255,255,0.05)"],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
      }],
    },
    options: {
      responsive: false,
      cutout: "75%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 1200, easing: "easeOutQuart" },
    },
  });
}


// ── Animated Counter ─────────────────────────────────────────
function animateCounter(el, target, duration = 1200) {
  const start = performance.now();
  const startVal = 0;
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(startVal + (target - startVal) * ease);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}


// ── Render Results ───────────────────────────────────────────
function getVerdictClass(verdict) {
  switch (verdict) {
    case "STRONG_MATCH":  return "verdict-strong";
    case "GOOD_MATCH":    return "verdict-good";
    case "PARTIAL_MATCH": return "verdict-partial";
    default:              return "verdict-weak";
  }
}
function getVerdictLabel(verdict) {
  switch (verdict) {
    case "STRONG_MATCH":  return "✅ Strong Match";
    case "GOOD_MATCH":    return "🔵 Good Match";
    case "PARTIAL_MATCH": return "⚠️ Partial Match";
    default:              return "❌ Weak Match";
  }
}

function getBarColor(score) {
  if (score >= 75) return "linear-gradient(90deg,#10b981,#34d399)";
  if (score >= 50) return "linear-gradient(90deg,#3b82f6,#60a5fa)";
  if (score >= 30) return "linear-gradient(90deg,#eab308,#facc15)";
  return "linear-gradient(90deg,#ef4444,#f87171)";
}

function renderResults(data) {
  // Score banner
  const scoreEl = document.getElementById("score-display");
  animateCounter(scoreEl, data.ats_score);
  renderGauge(data.ats_score);

  const verdictEl = document.getElementById("score-verdict");
  verdictEl.textContent  = getVerdictLabel(data.overall_verdict);
  verdictEl.className    = "score-verdict " + getVerdictClass(data.overall_verdict);

  document.getElementById("semantic-score-val").textContent = data.semantic_score?.toFixed(1) + "%";
  document.getElementById("keyword-score-val").textContent  = data.keyword_score?.toFixed(1) + "%";
  document.getElementById("resume-words-val").textContent   = (data.resume_length || "—") + " wds";

  // Summary
  document.getElementById("candidate-summary").textContent   = data.candidate_summary || "—";
  document.getElementById("verdict-explanation").textContent = data.verdict_explanation || "—";

  // Breakdown
  const breakdownEl = document.getElementById("breakdown-list");
  const bd = data.resume_score_breakdown || {};
  const labels = {
    experience_relevance: "Experience Relevance",
    skills_alignment:     "Skills Alignment",
    education_fit:        "Education Fit",
    achievements_impact:  "Achievements Impact",
  };
  breakdownEl.innerHTML = Object.entries(labels).map(([k, label]) => {
    const val = bd[k] ?? 0;
    return `
      <div class="breakdown-item">
        <div class="breakdown-label">
          <span>${label}</span>
          <span>${val}/100</span>
        </div>
        <div class="breakdown-bar">
          <div class="breakdown-fill" style="background:${getBarColor(val)}" data-width="${val}"></div>
        </div>
      </div>`;
  }).join("");

  // Animate bars
  requestAnimationFrame(() => {
    document.querySelectorAll(".breakdown-fill").forEach(fill => {
      fill.style.width = fill.dataset.width + "%";
    });
  });

  // Keywords
  const matchedEl = document.getElementById("matched-chips");
  const missingEl = document.getElementById("missing-chips");
  matchedEl.innerHTML = (data.matched_keywords?.length)
    ? data.matched_keywords.map(k => `<span class="chip chip-match">${k}</span>`).join("")
    : '<span class="chip chip-match">None detected</span>';
  missingEl.innerHTML = (data.missing_keywords?.length)
    ? data.missing_keywords.map(k => `<span class="chip chip-miss">${k}</span>`).join("")
    : '<span class="chip chip-match">All keywords covered!</span>';

  // Strengths
  const strengthsEl = document.getElementById("strengths-list");
  strengthsEl.innerHTML = (data.strengths?.length)
    ? data.strengths.map(s => `
        <div class="feature-item">
          <div class="feature-icon"><i class="fa-solid fa-check" style="color:var(--clr-green)"></i></div>
          <div class="feature-body">
            <div class="feature-title">${escHtml(s.title)}</div>
            <div class="feature-desc">${escHtml(s.description)}</div>
          </div>
        </div>`).join("")
    : "<p style='color:#64748b;font-size:.85rem'>—</p>";

  // Improvements
  const improvEl = document.getElementById("improvement-list");
  improvEl.innerHTML = (data.improvement_areas?.length)
    ? data.improvement_areas.map(a => `
        <div class="feature-item">
          <div class="feature-icon"><i class="fa-solid fa-arrow-up" style="color:var(--clr-orange)"></i></div>
          <div class="feature-body">
            <div class="feature-title" style="display:flex;align-items:center;gap:.5rem">
              ${escHtml(a.title)}
              <span class="priority-badge priority-${(a.priority || "MEDIUM").toLowerCase()}">${a.priority || "MEDIUM"}</span>
            </div>
            <div class="feature-desc">${escHtml(a.description)}</div>
          </div>
        </div>`).join("")
    : "<p style='color:#64748b;font-size:.85rem'>—</p>";

  // Quick Wins
  const qwEl = document.getElementById("quick-wins-grid");
  qwEl.innerHTML = (data.quick_wins?.length)
    ? data.quick_wins.map((w, i) => `
        <div class="quick-win-card">
          <span class="quick-win-num">${String(i + 1).padStart(2, "0")}</span>
          <span class="quick-win-text">${escHtml(w)}</span>
        </div>`).join("")
    : "<p style='color:#64748b;font-size:.85rem'>—</p>";

  // Talking Points
  const tpEl = document.getElementById("talking-points-list");
  tpEl.innerHTML = (data.interview_talking_points?.length)
    ? data.interview_talking_points.map(p => `<li>${escHtml(p)}</li>`).join("")
    : "<li>—</li>";

  // Rewrite
  document.getElementById("rewrite-suggestion").textContent = data.rewrite_suggestion || "—";

  // Show
  show("results-section");
  document.getElementById("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

function escHtml(str) {
  const d = document.createElement("div");
  d.textContent = str || "";
  return d.innerHTML;
}


// ── Form Submission ──────────────────────────────────────────
const form      = document.getElementById("analyze-form");
const submitBtn = document.getElementById("submit-btn");
const btnText   = submitBtn.querySelector(".btn-text");
const btnLoad   = submitBtn.querySelector(".btn-loading");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Validate
  if (!selectedFile) {
    showError("Please upload your resume (PDF or DOCX).");
    return;
  }
  const jd = textarea.value.trim();
  if (jd.length < 50) {
    showError("Job description must be at least 50 characters long.");
    return;
  }

  // UI: loading state
  submitBtn.disabled = true;
  btnText.classList.add("hidden");
  btnLoad.classList.remove("hidden");
  hide("results-section");
  hide("error-toast");
  showLoading();

  try {
    const formData = new FormData();
    formData.append("resume", selectedFile);
    formData.append("job_description", jd);

    const res = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || "Analysis failed. Please try again.");
    }

    hideLoading();
    renderResults(data);

  } catch (err) {
    hideLoading();
    showError(err.message || "An unexpected error occurred.");
    console.error(err);
  } finally {
    submitBtn.disabled = false;
    btnText.classList.remove("hidden");
    btnLoad.classList.add("hidden");
  }
});


// ── Re-analyze ───────────────────────────────────────────────
document.getElementById("reanalyze-btn").addEventListener("click", () => {
  hide("results-section");
  clearFile();
  textarea.value = "";
  charCount.textContent = "0";
  charCount.style.color = "";
  document.getElementById("analyzer").scrollIntoView({ behavior: "smooth" });
});


// ── Intersection Observer Animations ─────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animationPlayState = "running";
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll(".step-card, .stat-item").forEach(el => {
  el.style.animationPlayState = "paused";
  observer.observe(el);
});
