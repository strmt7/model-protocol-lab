# Security and data handling

## Trust boundaries

Treat checkpoints, tokenizer files, fixtures, generated text and reports as separate trust boundaries. Local-only loading and safetensors reduce exposure; they do not make arbitrary files, native dependencies or GPU drivers safe. Run unfamiliar model assets in an appropriately isolated environment with resource limits.

Use trusted, immutable checkpoint directories. Do not grant this process unnecessary filesystem or network access. The adapter requests local-only model resolution, disables remote model code and rejects pickle weight layouts; this has not been verified by runtime network-isolation tests.

## Data disclosure

No network reporting or telemetry is implemented in this package. Dependency behavior has not been audited. Default report JSON excludes raw observation arrays, but names, format strings, fingerprints, counts and decisions remain. Evidence mode retains answer text and input/output token IDs. Token IDs are not anonymization.

Do not publish actual prompts, outputs, datasets, weights, local paths, credentials, account configuration or dependency diagnostic logs without reviewing them. The source archive contains synthetic examples only. `.gitignore` is a convenience, not a secret scanner or access-control boundary.

The report renderer escapes displayed strings and uses a restrictive content-security policy. Browser behavior and hostile-content rendering still require testing. Do not expand it to render model output as HTML or execute supplied JavaScript.

## Operational limits

Current safeguards limit fixture size, candidate count, token output and configured context. They do not guarantee bounded memory, runtime or GPU utilization. Model loading can be expensive before evaluation starts. Cancellation does not preempt a running native operation.

Do not expose the library or CLI directly as a multi-user network service. Authentication, tenant isolation, safe job scheduling, audit logging and resource enforcement are not implemented.

## Reporting defects

For public issues, provide a minimal synthetic reproduction and dependency versions. Exclude credentials, proprietary prompts, private model files and personally identifying data. For sensitive defects, use a repository's enabled private security-reporting route when available; do not assume one has been configured.
