# Meissonic implementation plan

1. Add pinned published configuration and safetensors-header fixtures.
2. Add a native Meissonic transformer and scheduler with exact random-weight reference tests.
3. Add the backend, tiny published-layout snapshot helper, unit tests, and tiny end-to-end
   parity test.
4. Register the pinned catalog model and backend, then add real-checkpoint reference coverage.
5. Update the README, catalog/model/validation docs, validation grouping, notices, and license.
6. Run formatting, static checks, targeted reference tests, both required fast suites, and
   leave all worktree changes uncommitted.
