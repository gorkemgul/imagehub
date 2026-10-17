# Chroma1-Base implementation plan

1. Add a failing unit/reference test for Beta(0.6, 0.6) quantiles and flow-match beta schedules.
2. Implement the pure Python beta quantile and schedule conversion in the shared flow-match Euler
   scheduler, preserving the reference's float64 operations before its float32 tensor conversion.
3. Add `chroma1-base` to the catalog, Chroma unit tests and backend aliases with the pinned revision,
   shared file layout, 40-step default and guidance 3.0.
4. Add the published Base scheduler fixture and parameterize the Chroma reference suite over the Base
   snapshot, including exact scheduler, end-to-end and separate seed-42 records.
5. Update the README model row and Chroma catalog, model and validation documentation without changing
   generated validation data.
6. Run targeted unit/reference tests, every reference test using the shared scheduler, formatting,
   lint, typing, the fast suite and the clean-environment suite.
