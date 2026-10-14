# Ovis-Image implementation plan

1. Save the pinned Hub configs, tokenizer files, shard indexes, and safetensors headers.
2. Add an Ovis transformer with the published tensor layout and strict loading.
3. Add the local-snapshot backend, catalog entry, and registry selector.
4. Verify the published header and a tiny transformer execution, then add model documentation.
