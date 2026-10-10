import logging
import traceback
from typing import Any

import runpod

from asr import HypaAsrEngine, InputError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
)
logger = logging.getLogger("handler")

_engine: HypaAsrEngine | None = None


def get_engine() -> HypaAsrEngine:
    """Built on the first job instead of at import.

    Loading the weights at module scope holds the worker out of RunPod's pool
    for the whole load, and a failure there kills the container before anything
    can report it: the queue then fills with jobs that never run, stay
    `inQueue` forever, and leave `failed` at zero with no error anywhere. Doing
    it here puts both the wait and any failure inside a job, where RunPod
    records them.
    """
    global _engine
    if _engine is None:
        _engine = HypaAsrEngine()
    return _engine


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input") or {}
    if not isinstance(job_input, dict):
        return {"error": "Runpod job input must be an object"}

    try:
        return get_engine().transcribe(job_input)
    except InputError as exc:
        logger.warning("Invalid ASR input: %s", exc)
        return {"error": str(exc)}
    except Exception as exc:
        logger.error("ASR job failed: %s\n%s", exc, traceback.format_exc())
        return {"error": "ASR transcription failed"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
