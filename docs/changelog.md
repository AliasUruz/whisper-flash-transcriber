# Changelog

## 2025-09-19
- Stabilized the Silero VAD pipeline with shape validation, float32 normalization, and JSON logging to `logs/vad_failure.jsonl`.
- Stripped ANSI escape sequences from console output and disabled transformer progress colors by default.
- Simplified the ASR settings panel with a `Mostrar avancado` toggle that hides backend and quantization options until requested.
