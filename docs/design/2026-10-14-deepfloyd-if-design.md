# DeepFloyd IF design

`deepfloyd-if-i-xl` combines `DeepFloyd/IF-I-XL-v1.0` at
`c03d510e9b75bce9f9db5bb85148c1402ad7e694` with its stage-II upscaler
`DeepFloyd/IF-II-L-v1.0` at `609476ce702b2d94aff7d1f944dcc54d4f972901`.
Both repositories publish float32 safetensors; the gated DeepFloyd IF License is non-commercial.

The backend mirrors diffusers 0.40.0 `IFPipeline` then
`IFSuperResolutionPipeline`: one 77-token T5-XXL conditioning pass per stage, pixel-space
DDPM sampling with learned-range variance, then bilinear upscaling, noising at level 250, and
stage-II sampling to 256×256. Stage III, safety checking and watermarking are outside scope.

The existing T5 encoder, tokenizer and DDPM scheduler are reused. The shared U-Net adds the
published IF text-pooling augmentation, timestep class conditioning and IF residual settings.
Caption cleaning is not exposed: plain prompts use the reference's normal path without optional
`bs4` and `ftfy`. The snapshot stores stage II under `stage_ii/`.

Validation covers configuration/header layouts, DDPM dynamic thresholding and noising, random
U-Net parity, a tiny combined reference run, and real checkpoint components and end-to-end runs.
The large T5 reference result is saved and freed before native loading.
