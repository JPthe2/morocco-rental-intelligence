"""Speech-to-text (faster-whisper) and text-to-speech (Piper) — both fully
local/offline once their models are cached on first use. No API keys.
"""
import io
import re
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

import config

_whisper_model = None
_piper_voice = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(audio_path: str) -> str:
    model = get_whisper_model()
    segments, _info = model.transcribe(audio_path, beam_size=1)
    return " ".join(segment.text.strip() for segment in segments).strip()


def _ensure_piper_voice_files() -> Path:
    config.VOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.VOICE_MODELS_DIR / f"{config.PIPER_VOICE}.onnx"
    if not model_path.exists():
        from piper.download_voices import download_voice
        download_voice(config.PIPER_VOICE, config.VOICE_MODELS_DIR)
    return model_path


def get_piper_voice():
    global _piper_voice
    if _piper_voice is None:
        from piper import PiperVoice
        model_path = _ensure_piper_voice_files()
        _piper_voice = PiperVoice.load(model_path)
    return _piper_voice


def chunks_to_wav_bytes(chunks: Iterable) -> bytes:
    """Concatenate Piper AudioChunk-like objects (sample_rate, sample_width,
    sample_channels, audio_float_array) into a single 16-bit PCM WAV file.
    Pulled out of synthesize() so it's testable without loading a real voice model.
    """
    buffer = io.BytesIO()
    wav_writer = None
    for chunk in chunks:
        if wav_writer is None:
            wav_writer = wave.open(buffer, "wb")
            wav_writer.setnchannels(chunk.sample_channels)
            wav_writer.setsampwidth(chunk.sample_width)
            wav_writer.setframerate(chunk.sample_rate)
        pcm = (np.clip(chunk.audio_float_array, -1.0, 1.0) * 32767).astype(np.int16)
        wav_writer.writeframes(pcm.tobytes())
    if wav_writer is not None:
        wav_writer.close()
    return buffer.getvalue()


def strip_markdown_for_speech(text: str) -> str:
    """Agent replies are markdown (bold, bullet lists, tables, links) meant for
    on-screen text. Piper phonemizes whatever it's given literally, so feeding it
    raw markdown makes it try to pronounce '**', '|', '#', etc. — this is what
    produced the garbled, not-a-real-language audio before this was added."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)          # [text](url) -> text
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)                   # **bold**
    text = re.sub(r"__(.*?)__", r"\1", text)                       # __bold__
    text = re.sub(r"\*(.*?)\*", r"\1", text)                       # *italic*
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)     # # headers
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)   # bullet markers
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)   # numbered list markers
    text = text.replace("`", "").replace("|", " ")                 # inline code, table pipes
    text = re.sub(r"~\s*(?=\d)", "about ", text)                    # ~13,400 -> about 13,400
    text = re.sub(r"\n+", ". ", text)                               # line breaks -> sentence breaks
    text = re.sub(r"\.\s*\.", ".", text)                            # collapse doubled periods from blank lines
    return re.sub(r"\s+", " ", text).strip()


def synthesize(text: str) -> bytes:
    """Return a WAV file (bytes) spoken from the given text."""
    voice = get_piper_voice()
    clean_text = strip_markdown_for_speech(text)
    return chunks_to_wav_bytes(voice.synthesize(clean_text))
