---
title: API reference
---

# API reference

Everything public is importable from the top-level `imagehub` package.

```python
import imagehub
imagehub.__version__   # "{{ ih.version }}"
```

## `pipeline`

```python
pipeline(task, *, backend, options=None, registry=None, **backend_options) -> Pipeline
```

Create a pipeline. Construction never touches the network and never loads weights.

| Argument | Meaning |
| --- | --- |
| `task` | `"t2i"` (text to image) |
| `backend` | a catalog model id such as `"sd-v1-5"`, a family name such as `"flux"`, or an existing `Backend` |
| `options` | default generation settings applied to every request |
| `model` | *(backend option)* folder with the downloaded snapshot |
| `device` | *(backend option)* `"cpu"` (default), `"mps"`, `"cuda"` |
| `dtype` | *(backend option)* `torch.float32` (default), `torch.float16`, `torch.bfloat16` |

## `Pipeline`

| Member | Description |
| --- | --- |
| `pipe(prompt, **settings)` | run one request and return an `ImageResult` |
| `pipe.batch(prompts, **settings)` | run several requests in one model session |
| `pipe.map(prompts, **settings)` | lazily process an iterable with bounded memory |
| `await pipe.arun(prompt, **settings)` | run off the event loop |
| `pipe.load()` / `pipe.close()` | load or release the weights explicitly |
| `with pipe:` | load on entry, release on exit |

Settings passed in a call win over a request's own parameters, which win over the pipeline's `options`.
The accepted settings depend on the architecture (typically `steps`, `guidance`, `width`, `height`,
`negative_prompt`, `num_images`, `seed`, `latents`, and `image` for the models that take one). They
are validated before the model loads, and an unknown setting raises an error.

## Results

`ImageResult`

| Field | Description |
| --- | --- |
| `images` | tuple of `Image` |
| `metadata` | mapping describing the run: model id, revision, seed, steps, size, device, dtype |

`Image` holds an 8-bit RGB or RGBA image as a `(height, width, channels)` `uint8` tensor in `.tensor`.

| Member | Description |
| --- | --- |
| `width`, `height`, `channels` | dimensions |
| `save(path)` | write a PNG (no Pillow required) |
| `to_png()` | PNG bytes |
| `Image.open(path)`, `Image.from_png(data)` | read a PNG |
| `to_pil()` | convert to `PIL.Image.Image` (needs the `image` extra) |

## Discovery

```python
imagehub.available_models()          # {"t2i": ("amused-256", ..., "z-image-turbo")}
imagehub.available_backends("t2i")   # selectors accepted as `backend=`
spec = imagehub.model_info("sd-v1-5")
spec.repo_id, spec.revision, spec.license, spec.image_input
```

`ModelSpec` records, per model: `id`, `task`, `family`, `repo_id`, `revision` (the validated commit),
`files`, `license`, `gated`, `provenance` and `image_input`.

## Downloading

```python
from imagehub.pretrained import download_pretrained
download_pretrained(model_id, destination, *, revision=None, token=None, progress=None) -> Path
```

The same function backs `python -m imagehub.pretrained <model-id> <empty-directory> [--revision REF]`.
See [Downloading weights](guides/weights.md).

## Exceptions and warnings

Catch `ImageHubError` to handle any library-specific operational failure. More specific recovery is
available with `MissingDependencyError` for an unavailable optional dependency, `CheckpointError`
for missing or incompatible checkpoint files, `DownloadError` for download or verification
failures, `BackendBusyError` for invalid concurrent backend use, and `BackendContractError` when an
adapter violates the task contract. Argument validation can still raise standard Python exceptions.

Restricted checkpoints emit `ImageHubLicenseWarning`; configure it with Python's `warnings` filters
when needed. `ImagehubLicenseWarning` is retained as a compatibility alias.

## Extending

| Name | Purpose |
| --- | --- |
| `Backend`, `TextToImageBackend` | base classes: lazy loading, option validation, serialized inference |
| `Pipeline`, `TextToImagePipeline` | preprocess, forward, postprocess |
| `BackendRegistry`, `backend_registry` | map selectors to backend factories |
| `T2IRequest`, `Task` | request and task types |

See [Contributing](contributing.md) for how a new model plugs in.
