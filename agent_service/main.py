"""FastAPI service: the agent's brain. Wraps the manual tool-calling loop in agent.py."""
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import agent
import config
import memory
import subagents
import voice

app = FastAPI(title="Morocco Rental Agent Service", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., examples=["demo-session-1"])
    message: str = Field(..., examples=["I'm looking for a 2-bedroom under 6000 MAD in Rabat"])


class ChatResponse(BaseModel):
    response_text: str
    degraded: bool


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": config.LLM_PROVIDER,
        "model": config.LLM_MODEL,
        "llm_configured": bool(config.LLM_API_KEY),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    memory.expire_stale_sessions()
    result = agent.run_turn(req.session_id, req.message)
    return ChatResponse(**result)


class NLFilterRequest(BaseModel):
    query: str = Field(..., examples=["3-bedroom furnished apartments in Rabat with parking under 7000 MAD"])


@app.post("/nl-filter")
def nl_filter(req: NLFilterRequest):
    """Translate a natural-language search request into the structured filters
    the frontend's Listings tab already supports (city / max_price / min_bedrooms).
    A single one-shot LLM call, not the full agent loop — no memory, no tools."""
    return agent.extract_filters(req.query)


# ---------------------------------------------------------------------------
# Multi-agent orchestration: direct access to the specialist sub-agents, for
# debugging/demoing each one in isolation from the Concierge (agent.run_turn
# already delegates to these internally on every /chat and /voice/turn call).
# ---------------------------------------------------------------------------

class SubAgentRequest(BaseModel):
    question: str = Field(..., examples=["What's the average rent for a 2-bedroom in Rabat?"])


@app.post("/agents/scout")
def agents_scout(req: SubAgentRequest):
    return subagents.run_scout(req.question)


@app.post("/agents/analyst")
def agents_analyst(req: SubAgentRequest):
    return subagents.run_analyst(req.question)


# ---------------------------------------------------------------------------
# Voice (Phase 3)
# ---------------------------------------------------------------------------

class SpeakRequest(BaseModel):
    text: str = Field(..., examples=["The average rent in Rabat is 6500 MAD."])


class VoiceTurnResponse(BaseModel):
    transcript: str
    response_text: str
    audio_url: str | None
    degraded: bool


def _save_upload_to_temp(file: UploadFile) -> str:
    suffix = Path(file.filename or "audio").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        return tmp.name


@app.post("/voice/transcribe")
def voice_transcribe(file: UploadFile):
    tmp_path = _save_upload_to_temp(file)
    try:
        transcript = voice.transcribe(tmp_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"transcript": transcript}


@app.post("/voice/speak")
def voice_speak(req: SpeakRequest):
    try:
        wav_bytes = voice.synthesize(req.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {exc}") from exc
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/voice/turn", response_model=VoiceTurnResponse)
def voice_turn(file: UploadFile, session_id: str = "voice-default"):
    """One push-to-talk round trip: transcribe -> run the agent -> synthesize the
    reply. Returns transcript, text, and an audio URL together so the frontend
    can render text and play audio immediately, rather than waiting on separate
    round trips."""
    memory.expire_stale_sessions()
    tmp_path = _save_upload_to_temp(file)
    try:
        transcript = voice.transcribe(tmp_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not transcript:
        return VoiceTurnResponse(
            transcript="", response_text="I didn't catch anything — try recording again.",
            audio_url=None, degraded=True,
        )

    result = agent.run_turn(session_id, transcript)

    audio_url = None
    try:
        wav_bytes = voice.synthesize(result["response_text"])
        config.VOICE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.wav"
        (config.VOICE_OUTPUT_DIR / filename).write_bytes(wav_bytes)
        audio_url = f"/voice/audio/{filename}"
    except Exception:  # noqa: BLE001
        pass  # text response still returned even if TTS fails

    return VoiceTurnResponse(
        transcript=transcript,
        response_text=result["response_text"],
        audio_url=audio_url,
        degraded=result["degraded"],
    )


@app.get("/voice/audio/{filename}")
def voice_audio(filename: str):
    path = config.VOICE_OUTPUT_DIR / filename
    if ".." in filename or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(path, media_type="audio/wav")
