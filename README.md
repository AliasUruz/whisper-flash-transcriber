# Whisper Flash Transcriber

Whisper Flash Transcriber is a focused desktop companion for turning spoken thoughts into text without leaving your local machine. The default experience is intentionally small: press a hotkey, capture audio, run the CTranslate2 backend, and paste the result into the active application.

## Minimal workflow in four beats
1. **Toggle recording** with the primary hotkey (defaults to <kbd>F3</kbd>). The tray icon switches states immediately.
2. **Capture audio** until you toggle again. The recorder buffers to memory and only spills to disk when the advanced storage rules demand it.
3. **Transcribe through CTranslate2** using the curated Whisper model chosen at bootstrap. A single worker thread runs the inference pipeline.
4. **Paste the transcript** automatically (optional) or copy it to the clipboard so you can finish the edit wherever you are typing.

This cycle works out of the box on any machine that can install the base requirements.

## Quick start
1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/<your-account>/whisper-flash-transcriber.git
   cd whisper-flash-transcriber
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # PowerShell
   .\.venv\Scripts\activate
   # bash / zsh
   source .venv/bin/activate
   ```
3. Install the core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The lockfile ships with the CPU build of PyTorch so the baseline experience is identical across Windows, Linux, and macOS without touching custom package indexes.
4. Launch the tray application:
   ```bash
   python src/main.py
   ```
   The first boot provisions `config.json`, `secrets.json`, and `hotkey_config.json` under `~/.cache/whisper_flash_transcriber/` (or the directory pointed to `WHISPER_FLASH_PROFILE_DIR`). The hardware probe then recommends a Whisper model and starts the background download if one is missing.

### GPU acceleration and quantization (optional)
If you want CUDA-enabled PyTorch or advanced quantisation backends, install the base requirements first and then follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your driver. Extra GPU-centric packages (such as `bitsandbytes`) remain in `requirements-optional.txt` and should only be installed when you actually need them:
```bash
pip install -r requirements-optional.txt
```

## Configuration layout
`ConfigManager` keeps the canonical configuration at `~/.cache/whisper_flash_transcriber/config.json`. The schema is intentionally split between a minimal surface and a documented advanced tree.

### Top-level fields (always available)
| Key | Purpose |
| --- | --- |
| `record_key` | Global hotkey used to start/stop capture. |
| `record_mode` | `"toggle"` or `"press"` for hold-to-record. |
| `auto_paste` | Paste the final transcript into the foreground window. |
| `min_record_duration` | Ignore accidental taps shorter than this value (seconds). |
| `min_transcription_duration` | Drop segments that would produce negligible output. |
| `sound` | Enables the start/stop chime with frequency, duration, and volume controls. |
| `asr_model_id` | Whisper model served through the CTranslate2 runtime. |
| `asr_backend` | Runtime identifier (`"ctranslate2"` today). |
| `ui_language` | Two-letter locale for the settings window and tray menus. |

All other controls live inside the `advanced` namespace and mirror the structure exposed by the settings window.

### Advanced namespaces
| Path | Focus |
| --- | --- |
| `advanced.hotkeys` | Secondary shortcuts (agent mode key, auto-paste modifier overrides, keyboard library). |
| `advanced.ai` | Gemini/OpenRouter credentials, prompts, and timeouts for post-processing or agent mode. |
| `advanced.performance` | Batch sizing, device overrides, Torch compile toggle, and download parallelism. |
| `advanced.storage` | Recording retention, memory thresholds, and every directory override including `storage_root_dir`, `models_storage_dir`, `recordings_dir`, `asr_cache_dir`, `python_packages_dir`, `vad_models_dir`, and `hf_cache_dir`. |
| `advanced.vad` | Silero VAD enablement plus thresholds and padding in milliseconds. |
| `advanced.workflow` | UI niceties such as printing transcripts to the terminal. |
| `advanced.system` | OS integration flags like `launch_at_startup`.

The helpers exported by `src/config_schema.py` (`normalize_payload_tree`, `flatten_config_tree`, and `path_for_key`) keep the legacy flat keys working while the application migrates to the nested representation. Existing `config.json` files remain compatible: missing values are backfilled from the defaults above and misconfigured entries fall back gracefully during validation.

## Optional extras (`requirements-extras.txt`)
Install the extras only when you need the corresponding workflow. They are not required for the core hotkey → capture → CTranslate2 → paste loop.

```bash
pip install -r requirements-extras.txt
```

| Package | Enables |
| --- | --- |
| `google-generativeai` | Gemini-based correction and agent mode (`advanced.ai.text_correction_enabled = true`). |
| `onnxruntime` | Silero VAD acceleration (`advanced.vad.use_vad = true`). |
| `playwright` | The UI automation bridge used by integration scripts in `scripts/` and power-user workflows. |
| `accelerate`, `datasets[audio]` | Batch automation helpers and dataset tooling used by advanced users when remediating dependencies or benchmarking models. |

When you opt into any of these features, double-check the related configuration keys inside the `advanced` namespace so the runtime knows which services are active. Secrets continue to live in `secrets.json` even when the file is written in the new nested layout.

## Documentation map
| File | Summary |
| --- | --- |
| `docs/first-run.md` | How the onboarding wizard captures only the minimal information (hotkey, storage root, curated model) before deferring advanced tweaks to the main UI. |
| `docs/deployment.md` | Strategies for relocating the directories defined in `advanced.storage.*` and keeping cold starts short across machines. |
| `docs/dependency-audit.md` | Running the dependency audit against `requirements.txt`, `requirements-extras.txt`, and friends to detect missing packages quickly. |
| `docs/model-loading-flow.md` | Timeline for the CT2 model lifecycle and how the state manager keeps the tray icon accurate during downloads and reloads. |
| `docs/ui_vars.md` | Mapping between CustomTkinter variables and configuration keys, updated to reflect the advanced namespace. |

## Troubleshooting
- Use the built-in **Dependency Audit** panel (Settings → Diagnostics) to compare your environment with the manifests listed above. It reports missing packages, mismatched versions, and hash divergences.
- Logs live under `logs/` and can be tailed while running `python src/main.py` to observe hotkey detection, ASR reloads, and paste automation.
- The CLI helper `python -m compileall src` remains part of the pre-commit checklist to catch syntax errors quickly.

With the minimal workflow locked down, feel free to explore the advanced namespaces only when the project requirements grow beyond the default hotkey-driven transcription loop.
