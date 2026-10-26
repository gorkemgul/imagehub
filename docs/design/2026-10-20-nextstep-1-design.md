# NextStep-1 design

## Scope

Add native text-to-image inference for `nextstep-1-large`, the released
`stepfun-ai/NextStep-1-Large` checkpoint. Image editing and the pretraining and 1.1
checkpoints are out of scope.

## Published artifacts

- Weights: `stepfun-ai/NextStep-1-Large` at
  `97dd0b9e097f60e39e3ce3bbac1a3343a09547cd` (Apache-2.0).
- Reference code: `stepfun-ai/NextStep-1` at
  `a3303b62e11fc7cd473ba0e4b1f32eda73f4acc0` (Apache-2.0).
- The twelve bfloat16 safetensor shards contain a Qwen2.5-derived 14B decoder,
  image projectors, and a 12-block flow-matching MLP. The separately published
  VAE is a deterministic 16-channel KL autoencoder derived from the FLUX VAE.

## Native components

`NextStepModel` retains the published tensor names and implements Qwen2 grouped-query
causal attention, continuous image-token projection, patch packing, and the flow head.
The existing native KL autoencoder is reused with a key conversion for the VAE's legacy
checkpoint names. The existing Qwen tokenizer is reused.

Generation formats the prompt with the published image-area prefix, performs causal
autoregressive image-token generation, integrates each token with the published ODE
flow sampler, unpacks the latent grid, and decodes it. CPU noise generation and uint8
rounding follow the reference. Defaults are 256 by 256 pixels, CFG 1, and 20 flow steps.

## Memory and validation

The approximately 14B bfloat16 model needs about 29 GB for weights. Reference comparisons
therefore run in a separate process and release it before native loading. Offline tests cover
published headers, exact random-weight modules, prompt construction, option validation,
repeatability, and a tiny end-to-end model. Real-checkpoint tests use short 64-pixel,
one-flow-step generations for seeds 0 and 42; the published default remains unchanged.

