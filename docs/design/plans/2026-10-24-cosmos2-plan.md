# Cosmos-Predict2 2B implementation plan

1. Pin and fixture the published model index, component configs, tokenizer assets, weight index,
   and safetensors headers.
2. Add failing exact-parity tests for the Cosmos transformer and the required shared T5, Wan VAE,
   and Karras scheduler behavior.
3. Implement the native transformer and the minimal shared extensions, then prove random-weight
   component parity.
4. Add the backend, tiny published-layout snapshot tests, registry alias, and catalog entry.
5. Add opt-in real-checkpoint component and end-to-end tests using subprocess isolation.
6. Update the README, catalog, model, validation, license, and third-party documentation.
7. Format, lint, type-check, run all affected reference tests, run both fast suites, and build the
   local verified snapshot receipt after the foreground download completes.
