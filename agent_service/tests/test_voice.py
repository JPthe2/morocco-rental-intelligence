import sys
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import voice


@dataclass
class FakeChunk:
    sample_rate: int
    sample_width: int
    sample_channels: int
    audio_float_array: np.ndarray


def test_chunks_to_wav_bytes_produces_valid_wav_header():
    chunks = [FakeChunk(22050, 2, 1, np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32))]
    wav_bytes = voice.chunks_to_wav_bytes(chunks)
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


def test_chunks_to_wav_bytes_preserves_sample_rate_and_frame_count():
    samples = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)
    chunks = [FakeChunk(16000, 2, 1, samples)]
    wav_bytes = voice.chunks_to_wav_bytes(chunks)
    with wave.open(BytesIO(wav_bytes), "rb") as reader:
        assert reader.getframerate() == 16000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getnframes() == len(samples)


def test_chunks_to_wav_bytes_concatenates_multiple_chunks():
    chunk_a = FakeChunk(16000, 2, 1, np.array([0.1, 0.2], dtype=np.float32))
    chunk_b = FakeChunk(16000, 2, 1, np.array([0.3, 0.4, 0.5], dtype=np.float32))
    wav_bytes = voice.chunks_to_wav_bytes([chunk_a, chunk_b])
    with wave.open(BytesIO(wav_bytes), "rb") as reader:
        assert reader.getnframes() == 5


def test_chunks_to_wav_bytes_clips_out_of_range_samples():
    chunks = [FakeChunk(16000, 2, 1, np.array([2.0, -2.0], dtype=np.float32))]
    wav_bytes = voice.chunks_to_wav_bytes(chunks)
    with wave.open(BytesIO(wav_bytes), "rb") as reader:
        frames = reader.readframes(reader.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        assert samples[0] == 32767
        assert samples[1] == -32767


def test_chunks_to_wav_bytes_empty_input_returns_empty_bytes():
    assert voice.chunks_to_wav_bytes([]) == b""


def test_strip_markdown_removes_bold_and_bullets():
    text = "**Average Rent:** 8,500 MAD/month\n* Point one\n* Point two"
    cleaned = voice.strip_markdown_for_speech(text)
    assert "*" not in cleaned
    assert "Average Rent: 8,500 MAD/month" in cleaned
    assert "Point one" in cleaned and "Point two" in cleaned


def test_strip_markdown_removes_links_and_tables():
    text = "See [this listing](https://example.com/1) | Price | 5000 |"
    cleaned = voice.strip_markdown_for_speech(text)
    assert "http" not in cleaned
    assert "[" not in cleaned and "]" not in cleaned
    assert "|" not in cleaned
    assert "this listing" in cleaned


def test_strip_markdown_removes_headers_and_numbered_lists():
    text = "### Market Context\n1. First item\n2. Second item"
    cleaned = voice.strip_markdown_for_speech(text)
    assert "#" not in cleaned
    assert "Market Context" in cleaned
    assert "First item" in cleaned


def test_strip_markdown_expands_approx_tilde_before_numbers():
    cleaned = voice.strip_markdown_for_speech("Average Rent: ~13,400 MAD/month")
    assert "~" not in cleaned
    assert "about 13,400" in cleaned
