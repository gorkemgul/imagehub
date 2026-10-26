# DeepFloyd IF implementation plan

1. Save both pinned repository configurations, tokenizer assets and safetensors headers.
2. Extend the shared U-Net and DDPM scheduler only for IF configuration values.
3. Add the combined stage-I/stage-II backend, catalog entry and registry alias.
4. Add component, tiny reference and real-snapshot reference coverage.
5. Document the catalog, public settings and pending real-checkpoint validation.
