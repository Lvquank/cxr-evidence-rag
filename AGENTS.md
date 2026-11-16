# Experiment defaults

- For all future report-generation experiments, use `reasoning_effort="high"`
  with `gpt-5.6-luna`, unless the user explicitly requests another setting.
  This also applies to latency profiling and all four comparison branches.
  Do not silently substitute `none` to reduce latency.
- Use CUDA for local neural-model inference and clinical evaluators. Check GPU
  availability before starting; stop instead of silently falling back to CPU.
  Keep the frozen sklearn learned gate and existing retrieval protocol unchanged.
- Budget enough output tokens for reasoning plus report text. The default is
  8192 for high-effort report generation. Use identical effort and token caps
  across compared branches, and record the actual settings and API usage.
- Use new output directories, preserve previous runs, disable local response
  cache for a fresh experiment, and never relabel an old report as a new run.
- Validate all 206 studies and report provenance before recomputing metrics.
  Do not fabricate missing reference text; disclose the common clinical cohort.
