# Hypa Runpod Workers

Personal build repo for the two Hypa ASR Runpod worker images, so they can be pushed to a
GHCR namespace this account owns and pulled by a Runpod account without registry credentials.


Pushing to `main` builds both images for `linux/amd64` and pushes `:latest` and `:<sha>`.
Use **Run workflow** to rebuild without a code change.



Keeping this repository private and the packages public is deliberate: the repository holds
the source, the package is the only part Runpod needs. Note that a public package makes its
contents — including the worker source baked into the image — downloadable by anyone.

## Source of truth

