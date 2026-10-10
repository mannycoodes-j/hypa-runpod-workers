# Hypa ASR Streaming Runpod Worker

WebSocket streaming worker for `hypaai/Hypa-Whisper-Small-v2`.

This is the streaming sibling of `apps/hypa-asr-runpod-worker`. It is built for a Runpod **Load Balancer** endpoint, not a queue endpoint, because WebSocket connections need a direct long-lived route to the worker.

## Runtime

- Model: `hypaai/Hypa-Whisper-Small-v2`
- Protocol: WebSocket
- Route: `/ws`
- Health route: `/ping`
- Expected Runpod endpoint type: **Load Balancer**
- Exposed HTTP port: `80`

## Docker Image

The production image is built and published by GitHub Actions, not by hand. `.github/workflows/runpod-asr-stream.yml` builds and pushes it on every push that touches this worker, so the image is owned by the `hypaai` organization rather than by a personal registry.

```text
ghcr.io/hypaai/hypa-asr-stream-runpod-worker:latest
ghcr.io/hypaai/hypa-asr-stream-runpod-worker:<commit-sha>
```

Use the commit-sha tag when a Runpod endpoint should stay pinned to one exact revision; `latest` moves with every merge.

To build the same image locally for testing, without pushing it anywhere:

```bash
cd apps/hypa-asr-stream-runpod-worker

docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -t hypa-asr-stream-runpod-worker:local .
```

Keep both flags if you publish from a local build: Runpod pulls `linux/amd64`, and provenance attestations add an extra manifest to the index that some pullers reject.

## Runpod Settings

- Endpoint type: **Load Balancer**
- Exposed HTTP port: `80`
- GPU: start with a 16 GB GPU
- Env:

```text
HYPAAI_ASR_MODEL=hypaai/Hypa-Whisper-Small-v2
PORT=80
```

Add `HF_TOKEN` only if the model requires it.

## WebSocket URL

```text
wss://<endpoint-id>.api.runpod.ai/ws
```

Clients should include the Runpod API key as the endpoint authorization header when connecting directly to Runpod.

## Protocol

On connect, the worker sends:

```json
{
  "type": "connected",
  "message": "Transcription server ready",
  "model": "hypaai/Hypa-Whisper-Small-v2"
}
```

Send binary WAV chunks. The worker expects mono 16 kHz PCM WAV chunks, matching the existing Hypa Lab capture flow.

The worker responds with:

```json
{
  "type": "caption",
  "text": "transcribed text",
  "duration": 0.9,
  "processing_time": 0.18,
  "rtf": 0.2
}
```

Control messages:

```json
{ "type": "config", "temperature": 0.3, "top_p": 0.9, "beam_size": 1 }
{ "type": "ping" }
{ "type": "stop" }
```

The worker also accepts JSON base64 chunks:

```json
{
  "type": "audio_chunk",
  "audio": "BASE64_WAV_CHUNK"
}
```
