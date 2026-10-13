# PixArt-LCM implementation plan

1. Add published configuration and safetensors-header fixtures for the pinned checkpoint.
2. Teach the PixArt backend to select the checkpoint scheduler and pass its seeded generator to
   LCM sampling; allow the published LCM beta schedule.
3. Add the catalog entry, unit tests, tiny reference parity, and real-snapshot parametrization.
4. Document the checkpoint in the catalog, model guide, README, and pending validation section.
