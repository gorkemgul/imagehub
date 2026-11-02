---
name: add-or-validate-image-model
description: Add, complete or audit an imagehub image-generation model: catalog entry, download file list, native PyTorch modules, tokenizer, scheduler, backend, parity evidence, validation records and docs. Use whenever a model, checkpoint, tokenizer, scheduler, backend, registry entry or catalog row changes.
---

# Add or Validate an Image Model

## 1. Inventory

1. Read `.ai/AGENTS.md`, `.ai/GOAL.md` and `.ai/LOOP.md`.
2. Find the architecture in the awesome-text-to-image-architectures catalog (`data/models.json`)
   and note its `id` (this becomes `catalog_id`), weight license and variants.
3. For every released weight set, record from the Hub API (`/api/models/<repo>/revision/<ref>?blobs=true`):
   the repository, the immutable 40-character commit, whether the repository belongs to the
   authors or is a mirror (`provenance`), and the exact files needed for full-precision
   inference. Exclude reduced-precision copies, non-EMA weights, single-file duplicates,
   safety checkers and any Python files.
4. Read the published configs and the safetensors headers (HTTP range requests, no full
   download) to get every tensor name, dtype and shape. Check whether all variants share one
   layout.

## 2. Prototype outside the repository

1. Install the reference implementation (diffusers, transformers or the original repository)
   in a scratch environment.
2. Write the native modules and compare them with the reference using random weights, first
   on a tiny config and then at full size, including awkward input shapes. The target is an
   exact match (difference 0). If it is not exact, find out why before going further.
3. Compare the whole generation loop the same way (prompt encoding, guidance, noise, sampler
   steps, decoding, uint8 conversion).
4. Only verified code goes into the implementation plan.

### Unreleased reference implementations

When no released reference version contains a model, create a separate environment named
`.venv-ref-<model-id>` with the same torch and transformers versions as the main environment and
pin the upstream package to the exact commit that contains the implementation. Record that commit
as the model's reference metadata. Re-run parity against a released version when one includes the
model, and replace the commit reference with that release version.

## 3. Implement

- Model modules subclass `NativeModel`, keep the published tensor names, and convert keys only
  in a documented `prepare_state_dict` (for example dropping a constant buffer).
- Create non-persistent buffers so that they survive meta-device construction in
  `NativeModel.from_pretrained`; add a test that loads on meta and then moves to a device.
- Add the weight sets to `imagehub/catalog.py` with `catalog_id`, the pinned `revision`,
  `provenance`, the exact `files` list and the weight `license`. Register the family backend
  in `registry.py`; catalog ids become aliases automatically.
- The backend validates every option before loading, rejects unknown options with
  `TypeError`, and returns the task's result type with metadata (model id, revision, seed,
  sampler settings, device, dtype).
- Tokenizers must reproduce the reference exactly, including special-token handling (read the
  `normalized` flags from the tokenizer config) and pre-tokenizer splitting.

## 4. Parity ladder

Each rung is an opt-in reference test (`pytest.mark.reference`) that never downloads and
records its measurement through the `record` fixture before asserting.

1. **Header check** for every published snapshot: names and shapes match the native model on
   the meta device; record `missing`, `extra`, `mismatch` and tensor counts.
2. **Random-weight structural parity**, tiny and full size, against the reference library.
3. **Tokenizer / preprocessing parity** on fixed prompts plus a deterministic fuzz test over
   every real tokenizer.
4. **Scheduler / sampler parity**: timesteps exactly equal, and a multi-step trajectory with
   random inputs.
5. **Real-checkpoint component parity** on CPU float32: text encoder, denoiser at several
   timesteps, decoder and encoder.
6. **End-to-end parity** for every weight set with fixed initial noise: final latents or tokens
   and the uint8 image.
7. **Seed parity**: the same seed gives the same initial noise and image as the reference.
8. **Device smoke tests** (MPS, CUDA): finite output and correct shape; record the difference
   from CPU and the PSNR. No parity claim.

Also add unit tests (no network) with a tiny random snapshot written in the published layout:
generation shapes, option validation before loading, request isolation, repeatability with a
seed, and that the caller's autograd mode is restored.

## 5. Tolerances

A tolerance is 0 when the measured difference is 0. Otherwise it is the next power of ten above
the measured difference. Never pick a tolerance before measuring. A structural difference
(wrong layer order, wrong epsilon, wrong head count) is a bug, not noise: if a component differs
by more than about `1e-3`, or images by more than 2 pixel levels, debug instead of loosening.

## 6. Record and document

1. Run the reference suite on all snapshots with `IMAGEHUB_RECORD=docs/validation-measurements.json`.
2. Regenerate `docs/validation.json` with `scripts/build_validation.py`; every new model must
   end up `full` with an empty `tensor_header_check`.
3. Update `docs/validation.md`, the catalog table in `docs/catalog.md`, the model page in
   `docs/models.md` and the README model table. State limits explicitly (devices, dtypes,
   features not implemented).
4. Add the reference project to `THIRD_PARTY_NOTICES.md` and its license text to `licenses/`
   when code conventions were followed from it.
