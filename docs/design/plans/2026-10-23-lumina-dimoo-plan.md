# Lumina-DiMOO implementation plan

1. Record the pinned author configs and safetensors headers in fixtures.
2. Add a native LLaDA model retaining all published tensor names and exact random-weight parity.
3. Add cosine discrete sampling and a `lumina-dimoo` text-to-image backend using the shared
   tokenizer and VQ decoder; cover options, determinism, isolation, non-square output and tiny
   end-to-end parity.
4. Add the pinned catalog entry and registry mapping, reference tests for headers, tokenizer,
   model logits, VQ decoding and short real end-to-end runs.
5. Document the family in the README, catalog, model and validation pages; add attribution and
   validation grouping, then run formatting, typing, fast tests, clean-environment tests and all
   affected VQ reference tests.
