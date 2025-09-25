# Changelog

## 2025-02-14
- Replaced the invalid `torch==2.7.1` pin with the Windows-compatible `torch==2.5.1` CPU wheel and clarified GPU installation steps.
- Removed the conflicting `pydantic==2.7.1` constraint, keeping `pydantic>=2.9,<3` as the single policy across the project.
- Moved `bitsandbytes` to the optional dependency set with a platform guard so Windows installations no longer fail out of the box.

## 2025-09-19
- Stabilized the Silero VAD pipeline with shape validation, float32 normalization, and JSON logging to `logs/vad_failure.jsonl`.
- Stripped ANSI escape sequences from console output and disabled transformer progress colors by default.
- Simplified the ASR settings panel with a `Show advanced` toggle that hides backend and quantization options until requested.
