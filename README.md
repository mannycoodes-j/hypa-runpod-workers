# Hypa Runpod Workers

Personal build repo for the two Hypa ASR Runpod worker images, so they can be pushed to a
GHCR namespace this account owns and pulled by a Runpod account without registry credentials.

| Directory | Image | Endpoint type |
| -- | -- | -- |
| `hypa-asr-runpod-worker` | `ghcr.io/<owner>/hypa-asr-runpod-worker` | Serverless queue |
| `hypa-asr-stream-runpod-worker` | `ghcr.io/<owner>/hypa-asr-stream-runpod-worker` | Serverless load balancer (`/ws`) |

Pushing to `main` builds both images for `linux/amd64` and pushes `:latest` and `:<sha>`.
Use **Run workflow** to rebuild without a code change.

## These images must be public

Runpod pulls them without credentials, which only works if the **package** is public. GHCR
packages are private by default, so after the first successful build set each one to Public:

```
https://github.com/users/<owner>/packages/container/hypa-asr-runpod-worker/settings
https://github.com/users/<owner>/packages/container/hypa-asr-stream-runpod-worker/settings
```

Keeping this repository private and the packages public is deliberate: the repository holds
the source, the package is the only part Runpod needs. Note that a public package makes its
contents — including the worker source baked into the image — downloadable by anyone.

## Source of truth

The workers live in `hypaai/hypa-v2` under `apps/`. This repo is a build mirror: **change
the monorepo first**, then copy the directory here and push. Editing only here means the two
copies drift and the monorepo stops describing what is actually deployed.
