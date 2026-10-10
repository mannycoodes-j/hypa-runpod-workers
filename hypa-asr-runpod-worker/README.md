# Hypa ASR Runpod Worker

Runpod Serverless worker for `hypaai/Hypa-Whisper-Small-v2`.

This worker is intentionally a standard queue worker, not a WebSocket worker. It is meant for product ASR/STT requests from Hypa Translate: send one audio payload, receive one transcript.

## Runtime

- Model: `hypaai/Hypa-Whisper-Small-v2`
- Default task: transcription
- Input audio: base64, data URI, or URL
- Output: `{ "text": "...", "segments": [...], "model": "...", "timings": {...} }`

The model is loaded once, on the first job a worker handles, and reused by every job after it on that worker. Loading it at container startup instead would keep the worker out of the pool until the weights were in memory, and a failure during the load would kill the container before it could report one — leaving jobs stuck `inQueue` with nothing to explain why. The first job on a cold worker therefore pays the load; a network volume at `HF_HOME` makes that a disk read rather than a download.

## Input Shapes

Preferred Runpod input:

```json
{
  "input": {
    "audio": "BASE64_AUDIO",
    "audio_type": "base64",
    "mime_type": "audio/webm",
    "language": "en",
    "temperature": 0,
    "top_p": 0.5,
    "top_k": 1,
    "max_tokens": 400
  }
}
```

Compatibility input for the current Hypa API ASR caller:

```json
{
  "input": {
    "inputs": "BASE64_AUDIO",
    "stream": true,
    "parameters": {
      "generation_parameters": {
        "temperature": 0,
        "top_p": 0.5,
        "top_k": 1,
        "max_new_tokens": 400
      }
    }
  }
}
```

URL input:

```json
{
  "input": {
    "audio": "https://example.com/audio.wav",
    "audio_type": "url"
  }
}
```

## Runpod Endpoint URL

For queue-based Serverless, call:

```text
https://api.runpod.ai/v2/<endpoint-id>/runsync
```

Runpod wraps the handler return value under `output`, so callers should read `output.text`.

## Environment Variables

- `HYPAAI_ASR_MODEL`: Hugging Face model id. Defaults to `hypaai/Hypa-Whisper-Small-v2`.
- `HF_TOKEN`: optional Hugging Face token for private/gated models.
- `HF_HOME`: cache path. Defaults to `/runpod-volume/huggingface` in the Dockerfile so a network volume can reduce cold starts.
- `ASR_MAX_AUDIO_BYTES`: max downloaded/decoded input size before transcription. Defaults to 25 MB.
- `ASR_DEVICE`: optional override, `cuda` or `cpu`.

## Local Test

Create a `test_input.json` with one of the input shapes above, then run:

```bash
python handler.py
```

For a real local test, install `ffmpeg` and the Python requirements first. GPU is recommended for deployment; CPU works for smoke testing but is slow.
