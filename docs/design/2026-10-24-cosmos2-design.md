# Cosmos-Predict2 2B Text-to-Image design

## Scope

Add the published `nvidia/Cosmos-Predict2-2B-Text2Image` checkpoint as
`cosmos-predict2-2b-t2i`. The backend supports one text prompt, classifier-free guidance,
deterministic seeds, and CPU, CUDA, or MPS execution through the text-to-image API.

## Published snapshot

The author repository is pinned to `acdb5fde992a73ef0355f287977d002cbfd127e0`. It contains a
28-layer, 16-head Cosmos 3D diffusion transformer, an ungated ReLU T5 encoder, the Wan causal
VAE, and a Karras FlowMatch Euler schedule. The transformer and VAE are float32; the two T5
shards are bfloat16. Monolithic duplicate weights and the separately packaged tokenizer object
are excluded.

## Native components

- Add a Cosmos transformer whose module names match the published safetensors exactly.
- Reuse the native T5 encoder and tokenizer, extending T5 for its published ungated ReLU feed
  forward.
- Reuse the Wan VAE, supplying reference defaults omitted by this checkpoint's compact config.
- Reuse FlowMatch Euler, adding its published Karras schedule and sigma bounds.
- Add a backend matching `diffusers==0.40.0` `Cosmos2TextToImagePipeline`, including Cosmos
  preconditioning, the default negative prompt, single-frame latent layout, and Wan decoding.

## Parity and validation

Random-weight tests compare the transformer, shared T5 path, Wan defaults, scheduler, and a tiny
end-to-end pipeline exactly. Opt-in real-checkpoint tests cover tensor headers, tokenizer, text
encoder, denoiser, VAE, scheduler, seed 0 end to end, and a separate seed 42 end to end. The real
pipeline is loaded in a subprocess because the text encoder is large. Real end-to-end runs are
left to the validation controller.

## Memory and limits

The snapshot is roughly 13.4 GB on disk. CPU float32 inference expands the bfloat16 text encoder
and requires substantially more memory than the 2B transformer alone. Training, batches,
callbacks, safety-checker integration, image conditioning, and video generation are out of scope.
