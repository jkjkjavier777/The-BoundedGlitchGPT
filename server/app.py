import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from inference.generator import Generator
from server.schemas import GenerateRequest, GenerateResponse

CHECKPOINT = os.environ.get("BGGPT_CHECKPOINT", "checkpoints/model.pt")

state = {"generator": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the model once at startup, not on every request
    state["generator"] = Generator(CHECKPOINT)
    yield
    state["generator"] = None


app = FastAPI(title="The-BoundedGlitchGPT", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["generator"] is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    gen = state["generator"]
    if gen is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    text = gen.generate(
        req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    return GenerateResponse(text=text)
