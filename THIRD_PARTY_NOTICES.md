# Third-party architecture and license notices

imagehub implements native PyTorch inference for the model families listed below. These are new
implementations written with reference to upstream architecture definitions and checkpoint layouts;
they do not invoke the upstream inference packages at runtime. Layer structure, parameter names,
tokenizer rules and sampling order follow those projects so that published weights load unchanged
and outputs can be compared with them.

| Project | Attribution | Relevant imagehub components | License |
| --- | --- | --- | --- |
| [diffusers](https://github.com/huggingface/diffusers) v0.40.0 | The HuggingFace Team and contributors | U-Net, KL autoencoder, PNDM scheduler, SD sampling order, checkpoint layout | [Apache 2.0](licenses/Diffusers-Apache-2.0.txt) |
| [transformers](https://github.com/huggingface/transformers) v5.18.0 | The HuggingFace Team, OpenAI and contributors | CLIP text encoder, CLIP tokenizer normalization and splitting | [Apache 2.0](licenses/Transformers-Apache-2.0.txt) |
| [CLIP](https://github.com/openai/CLIP) | 2021 OpenAI | Byte-level BPE vocabulary conventions | [MIT](licenses/OpenAI-CLIP-MIT.txt) |

Model weights are not distributed with imagehub. Stable Diffusion 1.x weights are released under the
CreativeML OpenRAIL-M license, whose use restrictions bind whoever downloads and uses them.
