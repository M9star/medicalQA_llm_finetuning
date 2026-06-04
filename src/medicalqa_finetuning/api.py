"""FastAPI inference server for medical QA models."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import resolve_device
from .evaluate import extract_medmcqa_answer, extract_pubmedqa_answer
from .modeling import generate_response, load_model, load_tokenizer


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    question: str = Field(..., min_length=5, description="The medical question to answer.")
    task: Literal["medmcqa", "pubmedqa"] = Field("medmcqa", description="Task type.")
    options: dict[str, str] | None = Field(
        None,
        description='Multiple-choice options for medmcqa, e.g. {"A": "...", "B": "..."}.',
    )
    context: str | None = Field(None, description="Research context passage for pubmedqa.")
    max_new_tokens: int = Field(150, ge=10, le=512)


class PredictResponse(BaseModel):
    answer: str | None = Field(None, description="Extracted answer key (A/B/C/D or yes/no/maybe).")
    raw_response: str = Field(..., description="Full generated text from the model.")
    task: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    model_id: str
    adapter: str | None
    tasks: list[str]


# ---------------------------------------------------------------------------
# Prompt builders (Alpaca format, consistent with training)
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


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    model_id: str,
    adapter_path: Path | None = None,
    device: str = "auto",
    quantization: str = "auto",
) -> FastAPI:
    _state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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
        yield
        _state.clear()

    app = FastAPI(
        title="Medical QA API",
        description="Inference API for fine-tuned medical question-answering models.",
        version="0.1.0",
        lifespan=lifespan,
    )

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

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(req: PredictRequest):
        if "model" not in _state:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        if req.task == "medmcqa":
            prompt = _build_medmcqa_prompt(req.question, req.options)
            extractor = extract_medmcqa_answer
        else:
            prompt = _build_pubmedqa_prompt(req.question, req.context)
            extractor = extract_pubmedqa_answer

        raw = generate_response(_state["model"], _state["tokenizer"], prompt, req.max_new_tokens)
        answer = extractor(raw)
        return PredictResponse(answer=answer, raw_response=raw, task=req.task)

    return app
