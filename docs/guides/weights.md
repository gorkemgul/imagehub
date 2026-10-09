# Downloading weights

imagehub never downloads anything on its own. `pipeline()` only builds an object, and loading
reads local files. Fetching weights is a separate command you run on purpose.

```bash
python -m imagehub.pretrained <model-id> <empty-directory>
python -m imagehub.pretrained flux1-schnell checkpoints/flux1-schnell
```

or from Python:

```python
from imagehub.pretrained import download_pretrained

download_pretrained("sd-v1-5", "checkpoints/sd-v1-5")
```

## What the downloader guarantees

- The **revision is pinned** to the commit that passed the parity tests (see the
  [catalog](../catalog.md)). Asking for another revision works but is recorded as unverified.
- Every file is checked against its size. Weight files are also checked against the sha256 the Hub
  publishes; other files against their git blob hash.
- The destination must be empty, and an `imagehub_snapshot.json` manifest records exactly what was
  downloaded.
- Only HTTPS is accepted, and credentials are dropped on cross-host redirects.

## Gated repositories

Some repositories require you to accept terms first. Accept them on the model page on Hugging Face,
create a read token, and pass it through the environment:

```bash
export HF_TOKEN=...
python -m imagehub.pretrained krea2-turbo checkpoints/krea2-turbo
```

## Disk space

Checkpoints are large, often many gigabytes. Keep only the models you use; a snapshot
is an ordinary folder, so deleting it is all it takes to free the space.

## Licenses

imagehub is Apache-2.0, but **model weights are not**. Each has its own license, listed per model on
the [catalog](../catalog.md) and the model pages. Some are non-commercial, and some restrict
regions or uses. The license binds whoever downloads and uses the weights.
