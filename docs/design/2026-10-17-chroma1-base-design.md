# Chroma1-Base design

Chroma1-Base is another published checkpoint of the existing `chroma` family. The author repository
is `lodestones/Chroma1-Base`, pinned at revision
`50ebe125b67794f4ed8e4985d26a4414ca62aed3`, with Apache-2.0 weights and the same text encoder,
transformer, tokenizer and VAE layouts as Chroma1-HD and Chroma1-Flash. Its recommended settings are
40 inference steps and guidance 3.0.

The only runtime behavior missing from the existing family is the checkpoint's flow-match Euler
schedule. Its scheduler enables beta sigmas, which remap the shifted schedule between its existing
minimum and maximum through the inverse CDF of a symmetric Beta(0.6, 0.6) distribution. The runtime
implementation remains limited to Python, the standard library and torch. SciPy is a reference-test
dependency only.

The catalog entry reuses the existing Chroma file list. Reference coverage adds the Base snapshot to
the shared real-checkpoint parameters, retains the separate seed-42 generation comparison, and checks
the beta scheduler directly against diffusers and SciPy. The existing Chroma backend and model modules
do not change.
