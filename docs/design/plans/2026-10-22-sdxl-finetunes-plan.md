# Community SDXL fine-tunes implementation plan

1. Store pinned configs and safetensors tensor layouts for Juggernaut XL v9, Animagine XL 4.0 and
   DreamShaper XL 1.0.
2. Add the three full component snapshots to the catalog with immutable revisions, exact files,
   provenance, licenses and reference versions.
3. Extend the native DDPM and DEIS schedulers only for the spacing modes published by these
   checkpoints, then connect both existing schedulers to the SDXL backend.
4. Add catalog, scheduler, header, component and separate seed-0/seed-42 public-pipeline tests.
5. Update the README, catalog, model, validation and third-party documentation without changing
   generated validation records.
6. Download all three pinned snapshots, build verified local manifests, and run targeted checks,
   formatting, lint, typing, both required fast suites and all affected reference tests without
   running the long real-checkpoint tests.
