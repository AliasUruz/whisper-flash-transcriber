# Deployment notes

The application keeps its writable state outside of the source tree so that upgrades remain simple. Once the minimal workflow is configured, most deployments boil down to shipping a profile directory and pointing the runtime at it.

## Profile layout
All persistent files live under the profile directory (`~/.cache/whisper_flash_transcriber/` by default):

- `config.json` – runtime configuration. Top-level keys control the minimal workflow. Advanced knobs live under the `advanced` namespace.
- `secrets.json` – API keys for Gemini/OpenRouter (still referenced even when the main configuration is nested).
- `hotkey_config.json` – serialized representation of the active hotkeys.
- `logs/` – rotating logs produced by `logging_utils`.

Set the `WHISPER_FLASH_PROFILE_DIR` environment variable before launching `python src/main.py` to relocate the entire profile.

## Heavyweight directories
The following advanced fields point to directories that can grow several gigabytes. Keep them on fast local storage or pre-provision them on managed workstations.

| Key | Purpose |
| --- | --- |
| `advanced.storage.storage_root_dir` | Base directory. Other paths inherit from it unless explicitly overridden. |
| `advanced.storage.models_storage_dir` | Root for CTranslate2 models and metadata (`install.json`). |
| `advanced.storage.asr_cache_dir` | Raw model cache used by the ASR backend. |
| `advanced.storage.recordings_dir` | WAV recordings created during capture (temporary and persisted). |
| `advanced.storage.python_packages_dir` | Target for `pip install --target` when the dependency remediation flow installs wheels outside the virtual environment. Add it to `PYTHONPATH` when launching the app from scripts. |
| `advanced.storage.vad_models_dir` | Location of the Silero VAD files when `advanced.vad.use_vad` is enabled. |
| `advanced.storage.hf_cache_dir` | Shared Hugging Face cache for curated models. |

When migrating to a new machine:
1. Copy the profile directory and the advanced storage paths to the destination.
2. Update `config.json` if any of the absolute paths changed (a simple search-and-replace is often enough).
3. Launch the application once so it can recreate missing folders and validate access rights.

## Optional extras
The extras manifest (`requirements-extras.txt`) isolates packages that are not required for the minimal workflow. Install it only when the deployment actually uses the corresponding feature:

```bash
pip install -r requirements-extras.txt
```

- **Gemini/OpenRouter** – requires `google-generativeai` and a populated `secrets.json`.
- **Silero VAD** – requires `onnxruntime` and the VAD model in `advanced.storage.vad_models_dir`.
- **Automation scripts** – require `playwright`, plus the browsers installed via `playwright install`.
- **Dataset/benchmark helpers** – use `accelerate` and `datasets[audio]`.

If none of these workflows are active, skip the extras entirely to keep the footprint small.

## Adaptive performance tuning

The advanced performance profile now ships with self-adjusting heuristics for the CTranslate2 backend. When
`advanced.performance.batch_size_mode` and `advanced.performance.chunk_length_mode` are left in **auto**, the
runtime observes every inference run and keeps a short history of wall-clock duration and CUDA out-of-memory
signals. The feedback loop applies the following rules:

- **GPU workloads** start from the VRAM-aware baseline selected by `select_batch_size` and may grow towards the
  configured chunk length (up to ~1.6× the configured window). If the median transcription time is higher than the
  audio duration, both `batch_size` and `chunk_length_sec` are trimmed gradually until the system stabilises.
- **CPU workloads** keep a narrower window (up to ~20% above the configured chunk size) and prioritise keeping
  inference latency close to the recording length.
- **OOM events** immediately halve the current batch size and reduce the chunk length by ~40%. The new values are
  persisted in memory via `self.last_dynamic_batch_size` and reflected in the `[ASR] transcription_*` log entries.
- **Fast streaks** (median duration < ~65% of the chunk length) allow the engine to recover towards the configured
  targets, one gentle increment at a time, provided no OOM happened in the last three runs.

Operators who prefer fixed values can flip either mode to **manual**. Doing so resets the adaptive history and
forces the transcriber to honour the literal configuration on the next run.
