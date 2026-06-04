"""FastAPI inference server for medical QA models."""

from __future__ import annotations

from contextlib import asynccontextmanager
import random
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import DatasetConfig, resolve_device
from .data import load_prepared_dataset
from .modeling import generate_response, load_model, load_tokenizer


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    question: str = Field(..., min_length=5)
    task: Literal["medmcqa", "pubmedqa"] = "medmcqa"
    options: dict[str, str] | None = None
    context: str | None = None
    max_new_tokens: int = Field(150, ge=10, le=512)


class PredictResponse(BaseModel):
    answer: str | None
    raw_response: str
    task: str


class QuestionResponse(BaseModel):
    question: str
    options: dict[str, str] | None
    correct_answer: str
    subject: str | None
    context: str | None
    task: str


class ExplainRequest(BaseModel):
    question: str
    options: dict[str, str] | None = None
    correct_answer: str
    task: str
    context: str | None = None
    max_new_tokens: int = Field(300, ge=50, le=512)


class ExplainResponse(BaseModel):
    explanation: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    model_id: str
    adapter: str | None
    tasks: list[str]


# ---------------------------------------------------------------------------
# Dataset parsers
# ---------------------------------------------------------------------------

_MEDMCQA_ANS_RE = re.compile(r"The correct answer is ([A-D])\)")


def _parse_medmcqa_records(dataset) -> list[dict]:
    records = []
    for ex in dataset:
        input_text = ex.get("input", "")
        if "\n\nOptions:" not in input_text:
            continue
        question = input_text.split("\n\nOptions:")[0].replace("Question: ", "").strip()
        opts_text = input_text.split("Options:\n")[1].strip() if "Options:\n" in input_text else ""
        options: dict[str, str] = {}
        for line in opts_text.split("\n"):
            line = line.strip()
            if len(line) > 2 and line[1] == ")":
                options[line[0]] = line[3:].strip()
        m = _MEDMCQA_ANS_RE.search(ex.get("output", ""))
        correct = m.group(1) if m else None
        if question and len(options) == 4 and correct:
            records.append({
                "question": question,
                "options": options,
                "correct_answer": correct,
                "subject": ex.get("subject"),
                "context": None,
            })
    return records


def _parse_pubmedqa_records(dataset) -> list[dict]:
    records = []
    for ex in dataset:
        input_text = ex.get("input", "")
        question, context = "", ""
        if "Question:" in input_text:
            question = input_text.split("Question:")[-1].strip()
        if "Context:" in input_text:
            context = input_text.split("Context:")[1].split("\n\nQuestion:")[0].strip()
        output = ex.get("output", "").lower()
        correct = None
        for ans in ["yes", "no", "maybe"]:
            if output.startswith(ans):
                correct = ans
                break
        if question and correct:
            records.append({
                "question": question,
                "options": None,
                "correct_answer": correct,
                "subject": None,
                "context": context or None,
            })
    return records


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_medmcqa_prompt(question: str, options: dict[str, str] | None) -> str:
    instruction = (
        "You are a medical expert. Answer the following multiple-choice medical question. "
        "Choose the correct option and provide a brief explanation."
    )
    if options:
        opts = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
        input_text = f"Question: {question}\n\nOptions:\n{opts}"
    else:
        input_text = f"Question: {question}"
    return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"


def _build_pubmedqa_prompt(question: str, context: str | None) -> str:
    instruction = (
        "You are a medical expert. Based on the provided research context, answer "
        "the following yes/no/maybe question. Provide a brief explanation for your answer."
    )
    ctx = context or "No context provided."
    input_text = f"Context: {ctx}\n\nQuestion: {question}"
    return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"


def _build_explain_medmcqa_prompt(question: str, options: dict[str, str], correct_answer: str) -> str:
    opts = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
    correct_option = options.get(correct_answer, "")
    return (
        f"### Instruction:\nYou are a medical expert. Explain why option {correct_answer}) {correct_option} "
        f"is the correct answer to this question. Then briefly explain why each other option is incorrect.\n\n"
        f"### Input:\nQuestion: {question}\n\nOptions:\n{opts}\n\nCorrect Answer: {correct_answer}) {correct_option}\n\n"
        f"### Response:\n"
    )


def _build_explain_pubmedqa_prompt(question: str, context: str | None, correct_answer: str) -> str:
    ctx = context or "No context provided."
    return (
        f"### Instruction:\nYou are a medical expert. Explain why the answer to this research question "
        f"is '{correct_answer}'. Use the provided context to support your explanation.\n\n"
        f"### Input:\nContext: {ctx}\n\nQuestion: {question}\n\nCorrect Answer: {correct_answer}\n\n"
        f"### Response:\n"
    )


def _extract_answer(response: str, task: str) -> str | None:
    if task == "medmcqa":
        patterns = [
            r"correct answer is\s*([A-Da-d])",
            r"answer is\s*([A-Da-d])",
            r"^\s*([A-Da-d])\)",
            r"\b([A-Da-d])\)\s",
        ]
        for p in patterns:
            m = re.search(p, response, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        for ch in response[:50]:
            if ch.upper() in {"A", "B", "C", "D"}:
                return ch.upper()
    else:
        rl = response.lower().strip()
        for ans in ["yes", "no", "maybe"]:
            if rl.startswith(ans):
                return ans
        for pattern, answer in [(r"\byes[.,\s]", "yes"), (r"\bno[.,\s]", "no"), (r"\bmaybe[.,\s]?", "maybe")]:
            if re.search(pattern, rl):
                return answer
    return None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    model_id: str,
    adapter_path: Path | None = None,
    device: str = "auto",
    quantization: str = "auto",
    prepared_data_dir: Path | None = None,
) -> FastAPI:
    _state: dict = {}
    _data_dir = prepared_data_dir or DatasetConfig().output_dir

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load model
        device_info = resolve_device(device, quantization)
        tokenizer = load_tokenizer(model_id)
        model = load_model(model_id, device_info)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.eval()
        _state["model"] = model
        _state["tokenizer"] = tokenizer
        _state["model_id"] = model_id
        _state["adapter_path"] = str(adapter_path) if adapter_path else None

        # Load quiz questions from prepared datasets
        medmcqa_path = _data_dir / "medmcqa_alpaca"
        pubmedqa_path = _data_dir / "pubmedqa_alpaca"
        if medmcqa_path.exists():
            dd = load_prepared_dataset(medmcqa_path)
            _state["questions_medmcqa"] = _parse_medmcqa_records(dd["train"])
        if pubmedqa_path.exists():
            dd = load_prepared_dataset(pubmedqa_path)
            _state["questions_pubmedqa"] = _parse_pubmedqa_records(dd["train"])

        yield
        _state.clear()

    app = FastAPI(
        title="Medical QA",
        description="Quiz and inference API for fine-tuned medical question-answering models.",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, tags=["ui"])
    def ui():
        return HTMLResponse(content=_HTML_UI, headers={"Cache-Control": "no-store"})

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health():
        return {"status": "ok", "model_loaded": "model" in _state}

    @app.get("/info", response_model=InfoResponse, tags=["meta"])
    def info():
        if "model" not in _state:
            raise HTTPException(status_code=503, detail="Model not yet loaded.")
        return {
            "model_id": _state["model_id"],
            "adapter": _state["adapter_path"],
            "tasks": ["medmcqa", "pubmedqa"],
        }

    @app.get("/question", response_model=QuestionResponse, tags=["quiz"])
    def get_question(task: str = Query("medmcqa")):
        key = f"questions_{task}"
        questions = _state.get(key, [])
        if not questions:
            raise HTTPException(status_code=404, detail=f"No questions loaded for task '{task}'.")
        q = random.choice(questions)
        return QuestionResponse(task=task, **q)

    @app.post("/explain", response_model=ExplainResponse, tags=["quiz"])
    def explain(req: ExplainRequest):
        if "model" not in _state:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        if req.task == "medmcqa":
            prompt = _build_explain_medmcqa_prompt(req.question, req.options or {}, req.correct_answer)
        else:
            prompt = _build_explain_pubmedqa_prompt(req.question, req.context, req.correct_answer)
        raw = generate_response(_state["model"], _state["tokenizer"], prompt, req.max_new_tokens)
        return ExplainResponse(explanation=raw)

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(req: PredictRequest):
        if "model" not in _state:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        if req.task == "medmcqa":
            prompt = _build_medmcqa_prompt(req.question, req.options)
        else:
            prompt = _build_pubmedqa_prompt(req.question, req.context)
        raw = generate_response(_state["model"], _state["tokenizer"], prompt, req.max_new_tokens)
        answer = _extract_answer(raw, req.task)
        return PredictResponse(answer=answer, raw_response=raw, task=req.task)

    return app


# ---------------------------------------------------------------------------
# HTML Quiz UI
# ---------------------------------------------------------------------------

_HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Medical QA Quiz</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f0f4f8;min-height:100vh;display:flex;align-items:flex-start;justify-content:center;padding:32px 16px}
.wrap{width:100%;max-width:720px}
h1{font-size:1.6rem;color:#1a202c;margin-bottom:4px}
.subtitle{color:#718096;font-size:.9rem;margin-bottom:24px}
.toolbar{display:flex;gap:12px;margin-bottom:20px;align-items:center;flex-wrap:wrap}
select{border:1px solid #cbd5e0;border-radius:8px;padding:9px 12px;font-size:.9rem;color:#2d3748;background:#fff;cursor:pointer;outline:none}
select:focus{border-color:#4299e1}
.btn-next{background:#4299e1;color:#fff;border:none;border-radius:8px;padding:9px 20px;font-size:.9rem;font-weight:600;cursor:pointer;transition:background .2s}
.btn-next:hover{background:#3182ce}
.btn-next:disabled{background:#a0aec0;cursor:not-allowed}
.card{background:#fff;border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,.07);padding:28px;margin-bottom:16px}
.meta{font-size:.78rem;color:#a0aec0;margin-bottom:12px;text-transform:uppercase;letter-spacing:.05em}
.question{font-size:1.05rem;color:#2d3748;line-height:1.6;margin-bottom:20px;font-weight:500}
.context-toggle{font-size:.82rem;color:#4299e1;cursor:pointer;margin-bottom:12px;display:inline-block}
.context-box{background:#f7fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;font-size:.85rem;color:#4a5568;line-height:1.6;margin-bottom:16px;display:none}
.options{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.opt-btn{border:2px solid #e2e8f0;border-radius:10px;padding:12px 14px;font-size:.9rem;color:#2d3748;background:#fff;cursor:pointer;text-align:left;transition:all .15s;display:flex;gap:10px;align-items:flex-start}
.opt-btn:hover:not(:disabled){border-color:#4299e1;background:#ebf8ff}
.opt-btn:disabled{cursor:not-allowed}
.opt-btn .key{font-weight:700;color:#4299e1;flex-shrink:0;min-width:18px}
.opt-btn.correct{border-color:#48bb78;background:#f0fff4}
.opt-btn.correct .key{color:#48bb78}
.opt-btn.wrong{border-color:#fc8181;background:#fff5f5}
.opt-btn.wrong .key{color:#fc8181}
.opt-btn.reveal{border-color:#48bb78;background:#f0fff4}
.opt-btn.reveal .key{color:#48bb78}
.yesno{display:flex;gap:10px}
.yn-btn{flex:1;border:2px solid #e2e8f0;border-radius:10px;padding:14px;font-size:1rem;font-weight:600;color:#2d3748;background:#fff;cursor:pointer;transition:all .15s}
.yn-btn:hover:not(:disabled){border-color:#4299e1;background:#ebf8ff}
.yn-btn:disabled{cursor:not-allowed}
.yn-btn.correct{border-color:#48bb78;background:#f0fff4;color:#276749}
.yn-btn.wrong{border-color:#fc8181;background:#fff5f5;color:#9b2c2c}
.yn-btn.reveal{border-color:#48bb78;background:#f0fff4;color:#276749}
.result-bar{display:none;padding:14px 18px;border-radius:10px;margin-top:16px;font-weight:600;font-size:1rem}
.result-bar.correct{background:#f0fff4;border:1px solid #9ae6b4;color:#276749}
.result-bar.wrong{background:#fff5f5;border:1px solid #fed7d7;color:#9b2c2c}
.explain-section{display:none;margin-top:16px}
.btn-explain{background:#805ad5;color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:.9rem;font-weight:600;cursor:pointer;transition:background .2s}
.btn-explain:hover{background:#6b46c1}
.btn-explain:disabled{background:#a0aec0;cursor:not-allowed}
.explanation{margin-top:14px;background:#faf5ff;border:1px solid #e9d8fd;border-radius:10px;padding:16px;font-size:.88rem;color:#44337a;line-height:1.7;white-space:pre-wrap;display:none}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-card{background:#fff;border-radius:12px;padding:28px;text-align:center;color:#a0aec0;box-shadow:0 2px 16px rgba(0,0,0,.07)}
</style>
</head>
<body>
<div class="wrap">
  <h1>Medical QA Quiz</h1>
  <p class="subtitle">Test your medical knowledge — AI explains every answer.</p>

  <div class="toolbar">
    <select id="task" onchange="switchTask()">
      <option value="medmcqa">MedMCQA — Multiple Choice</option>
      <option value="pubmedqa">PubMedQA — Yes / No / Maybe</option>
    </select>
    <button class="btn-next" id="btnNext" onclick="loadQuestion()">Next Question</button>
  </div>

  <div id="qcard" class="loading-card">Loading question...</div>
</div>

<script>
let _current = null;

async function switchTask() {
  await loadQuestion();
}

async function loadQuestion() {
  const task = document.getElementById('task').value;
  document.getElementById('btnNext').disabled = true;
  document.getElementById('qcard').innerHTML = '<div class="loading-card">Loading question...</div>';
  document.getElementById('qcard').className = 'loading-card';

  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 8000);
    const res = await fetch('/question?task=' + task + '&_=' + Date.now(),
                            {cache: 'no-store', signal: ctrl.signal});
    clearTimeout(timer);
    if (!res.ok) throw new Error(await res.text());
    _current = await res.json();
    renderQuestion(_current);
  } catch(e) {
    const msg = e.name === 'AbortError'
      ? 'Connection timed out (stale connection). Click Next Question to retry.'
      : 'Failed to load: ' + e.message;
    document.getElementById('qcard').innerHTML =
      '<div class="loading-card" style="color:#fc8181">' + msg + '</div>';
  }
  document.getElementById('btnNext').disabled = false;
}

function renderQuestion(q) {
  let html = '<div class="card">';
  if (q.subject) html += '<div class="meta">' + q.subject + '</div>';
  html += '<div class="question">' + escHtml(q.question) + '</div>';

  if (q.task === 'pubmedqa') {
    if (q.context) {
      html += '<span class="context-toggle" onclick="toggleCtx()">▶ Show research context</span>';
      html += '<div class="context-box" id="ctxBox">' + escHtml(q.context) + '</div>';
    }
    html += '<div class="yesno">';
    for (const ans of ['yes','no','maybe']) {
      html += '<button class="yn-btn" id="yn_'+ans+'" onclick="selectYN(\''+ans+'\')">' + cap(ans) + '</button>';
    }
    html += '</div>';
  } else {
    html += '<div class="options">';
    for (const [k,v] of Object.entries(q.options || {})) {
      html += '<button class="opt-btn" id="opt_'+k+'" onclick="selectOpt(\''+k+'\')">'
            + '<span class="key">'+k+'</span><span>'+escHtml(v)+'</span></button>';
    }
    html += '</div>';
  }

  html += '<div class="result-bar" id="resultBar"></div>';
  html += '</div>';

  // explain section
  html += '<div class="explain-section" id="explainSection">'
        + '<button class="btn-explain" id="btnExplain" onclick="getExplanation()">Get AI Explanation</button>'
        + '<div class="explanation" id="explanationBox"></div>'
        + '</div>';

  document.getElementById('qcard').className = '';
  document.getElementById('qcard').innerHTML = html;
}

function selectOpt(chosen) {
  const correct = _current.correct_answer;
  document.querySelectorAll('.opt-btn').forEach(b => b.disabled = true);

  document.getElementById('opt_' + chosen).classList.add(chosen === correct ? 'correct' : 'wrong');
  if (chosen !== correct) {
    document.getElementById('opt_' + correct).classList.add('reveal');
  }

  const bar = document.getElementById('resultBar');
  bar.style.display = 'block';
  if (chosen === correct) {
    bar.className = 'result-bar correct';
    bar.textContent = '✓ Correct!  Answer: ' + correct + ') ' + (_current.options[correct] || '');
  } else {
    bar.className = 'result-bar wrong';
    bar.textContent = '✗ Wrong.  Correct answer: ' + correct + ') ' + (_current.options[correct] || '');
  }
  document.getElementById('explainSection').style.display = 'block';
}

function selectYN(chosen) {
  const correct = _current.correct_answer;
  document.querySelectorAll('.yn-btn').forEach(b => b.disabled = true);
  document.getElementById('yn_' + chosen).classList.add(chosen === correct ? 'correct' : 'wrong');
  if (chosen !== correct) document.getElementById('yn_' + correct).classList.add('reveal');

  const bar = document.getElementById('resultBar');
  bar.style.display = 'block';
  if (chosen === correct) {
    bar.className = 'result-bar correct';
    bar.textContent = '✓ Correct!  Answer: ' + cap(correct);
  } else {
    bar.className = 'result-bar wrong';
    bar.textContent = '✗ Wrong.  Correct answer: ' + cap(correct);
  }
  document.getElementById('explainSection').style.display = 'block';
}

async function getExplanation() {
  const btn = document.getElementById('btnExplain');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Generating explanation...';

  const body = {
    question: _current.question,
    options: _current.options,
    correct_answer: _current.correct_answer,
    task: _current.task,
    context: _current.context,
    max_new_tokens: 300,
  };

  try {
    const res = await fetch('/explain', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const data = await res.json();
    const box = document.getElementById('explanationBox');
    box.textContent = data.explanation;
    box.style.display = 'block';
    btn.style.display = 'none';
  } catch(e) {
    btn.textContent = 'Get AI Explanation';
    btn.disabled = false;
  }
}

function toggleCtx() {
  const box = document.getElementById('ctxBox');
  const tog = document.querySelector('.context-toggle');
  if (box.style.display === 'block') {
    box.style.display = 'none'; tog.textContent = '▶ Show research context';
  } else {
    box.style.display = 'block'; tog.textContent = '▼ Hide research context';
  }
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

loadQuestion();
</script>
</body>
</html>
"""
