import base64
import binascii
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger("hypa_asr")

SAMPLE_RATE = 16_000
DEFAULT_MODEL_ID = "hypaai/Hypa-Whisper-Small-v2"
DATA_URI_MARKER = ";base64,"


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class AudioPayload:
    audio_bytes: bytes
    source: str
    mime_type: str | None = None


def _max_audio_bytes() -> int:
    return int(os.getenv("ASR_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))


def _strip_data_uri(value: str) -> tuple[str, str | None]:
    if not value.startswith("data:") or "," not in value:
        return value, None

    header, data = value.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or None
    if DATA_URI_MARKER not in header and ";base64" not in header:
        raise InputError("Only base64 data URIs are supported")
    return data, mime_type


def _decode_base64(value: str) -> bytes:
    data, _mime_type = _strip_data_uri(value.strip())
    try:
        return base64.b64decode(data, validate=True)
    except binascii.Error as exc:
        raise InputError(f"Invalid base64 audio: {exc}") from exc


def _download_audio(url: str) -> bytes:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    audio_bytes = response.content
    if len(audio_bytes) > _max_audio_bytes():
        raise InputError("Audio download exceeds ASR_MAX_AUDIO_BYTES")
    return audio_bytes


def resolve_audio(job_input: dict[str, Any]) -> AudioPayload:
    audio = job_input.get("audio") or job_input.get("audio_file") or job_input.get("inputs")
    if not isinstance(audio, str) or not audio.strip():
        raise InputError("Input must include audio, audio_file, or inputs")

    audio_type = (job_input.get("audio_type") or "").strip().lower()
    mime_type = job_input.get("mime_type")
    if not isinstance(mime_type, str):
        mime_type = None

    if audio_type == "url" or audio.startswith(("http://", "https://")):
        audio_bytes = _download_audio(audio)
        return AudioPayload(audio_bytes=audio_bytes, source="url", mime_type=mime_type)

    data, data_uri_mime = _strip_data_uri(audio)
    audio_bytes = _decode_base64(data)
    if len(audio_bytes) > _max_audio_bytes():
        raise InputError("Audio input exceeds ASR_MAX_AUDIO_BYTES")
    return AudioPayload(
        audio_bytes=audio_bytes,
        source="base64",
        mime_type=mime_type or data_uri_mime,
    )


def decode_audio(audio_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".audio") as input_file:
        input_file.write(audio_bytes)
        input_file.flush()

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_file.name,
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise InputError(f"Could not decode audio with ffmpeg: {detail}") from exc

    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise InputError("Decoded audio is empty")
    return audio


def _device() -> str:
    override = os.getenv("ASR_DEVICE", "").strip().lower()
    if override in {"cuda", "cpu"}:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def _generation_value(
    job_input: dict[str, Any],
    key: str,
    default: int | float | str | None,
) -> Any:
    generation = job_input.get("parameters", {}).get("generation_parameters", {})
    if key in job_input:
        return job_input[key]
    if isinstance(generation, dict) and key in generation:
        return generation[key]
    if key == "max_tokens" and isinstance(generation, dict):
        return generation.get("max_new_tokens", default)
    return default


def build_generate_kwargs(job_input: dict[str, Any]) -> dict[str, Any]:
    temperature = float(_generation_value(job_input, "temperature", 0) or 0)
    kwargs: dict[str, Any] = {
        "task": "transcribe",
        "temperature": temperature,
        "do_sample": temperature > 0,
        "num_beams": int(_generation_value(job_input, "num_beams", 1) or 1),
        "max_new_tokens": int(_generation_value(job_input, "max_tokens", 400) or 400),
    }

    language = job_input.get("language")
    if isinstance(language, str) and language.strip() and language.lower() != "auto":
        kwargs["language"] = language.strip()

    top_p = _generation_value(job_input, "top_p", None)
    if temperature > 0 and top_p is not None:
        kwargs["top_p"] = float(top_p)

    return kwargs


def _float_input(job_input: dict[str, Any], key: str, default: float) -> float:
    value = job_input.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class HypaAsrEngine:
    def __init__(self) -> None:
        self.model_id = os.getenv("HYPAAI_ASR_MODEL", DEFAULT_MODEL_ID)
        self.device_name = _device()
        self.torch_dtype = torch.float16 if self.device_name == "cuda" else torch.float32
        device_index = 0 if self.device_name == "cuda" else -1

        logger.info("Loading ASR model %s on %s", self.model_id, self.device_name)
        start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=os.getenv("HF_TOKEN") or None,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=os.getenv("HF_TOKEN") or None,
        )
        if self.device_name == "cuda":
            model.to("cuda")
        model.eval()

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=device_index,
        )
        logger.info("ASR model ready in %.2fs", time.perf_counter() - start)

    def transcribe(self, job_input: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        payload = resolve_audio(job_input)
        audio = decode_audio(payload.audio_bytes)
        decode_finished_at = time.perf_counter()

        return_timestamps = bool(job_input.get("return_timestamps", False))
        result = self.pipe(
            {"array": audio, "sampling_rate": SAMPLE_RATE},
            generate_kwargs=build_generate_kwargs(job_input),
            chunk_length_s=_float_input(job_input, "chunk_length_s", 30.0),
            return_timestamps=return_timestamps,
        )
        finished_at = time.perf_counter()

        text = str(result.get("text", "")).strip()
        response: dict[str, Any] = {
            "text": text,
            "model": self.model_id,
            "source": payload.source,
            "duration_seconds": round(float(audio.shape[0]) / SAMPLE_RATE, 3),
            "timings": {
                "decode_seconds": round(decode_finished_at - started_at, 3),
                "inference_seconds": round(finished_at - decode_finished_at, 3),
                "total_seconds": round(finished_at - started_at, 3),
            },
        }

        if return_timestamps and result.get("chunks"):
            response["segments"] = result["chunks"]

        return response
