# FLUX.1-Lite implementation plan

1. Record the pinned Hub configs, indexes and transformer tensor header fixture.
2. Add `flux-1-lite-8b` to the FLUX catalog with model-card defaults and exact component files.
3. Extend FLUX catalog/default and published-layout tests for the reduced eight-block transformer.
4. Add a dedicated opt-in real-checkpoint suite using subprocess reference generation for the two
   end-to-end seeds.
5. Update the README table, catalog/model/validation documentation, validation grouping and third-
   party notice.
6. Download the pinned local snapshot, build its manifest, and run formatting, static checks, fast
   tests, clean-environment tests and all applicable FLUX reference tests that do not require real
   weights.
