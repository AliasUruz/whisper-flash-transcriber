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
   pip install -r requirements.txt -c constraints.txt
   ```
   The base dependency set installs the faster-whisper/CTranslate2 runtime, `huggingface_hub`, and all core libraries needed for
   the default flow. Version specifiers now use bounded ranges (for example, `numpy>=1.26,<3` and `psutil>=5.9,<6.1`) so that
   upgrades remain compatible with PyTorch 2.5.1 and other vendor wheels without forcing exact pins. PyTorch and the
   Transformers pipeline are no longer part of the mandatory stack.

4. **Optional legacy stack:**
   ```bash
   pip install -r requirements-legacy.txt -c constraints.txt
   ```
   Install this file only if you plan to run the legacy Transformers + PyTorch workflow. The main application no longer ships
   that backend, but the optional dependencies remain available for custom forks. The file now pins `torch==2.5.1` to maintain
   parity with the production baseline; `huggingface_hub` ships with the core requirements, so no additional extras are pulled
   in by this step. When installing PyTorch on Linux without CUDA, use the CPU index explicitly:

   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
   ```

### Optional: Advanced GPU and quantization extras

Install the optional requirements only when you need GPU-centric optimizations such as quantized models:

### GPU acceleration and quantization (optional)
If you want CUDA-enabled PyTorch or advanced quantisation backends, install the base requirements first and then follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your driver. Extra GPU-centric packages (such as `bitsandbytes`) remain in `requirements-optional.txt` and should only be installed when you actually need them:
```bash
pip install -r requirements-optional.txt -c constraints.txt
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

### Headless Mode

When you need background operation without the system tray icon or any Tk windows,
launch Whisper Flash Transcriber in headless mode:

```bash
python src/main.py --headless
```

The recorder, transcription pipeline, hotkeys, and automatic paste continue to work,
but all Tk-based UI surfaces (settings window, onboarding wizard, diagnostic dialogs)
are suppressed. Any message box that would normally appear is logged instead so you
can monitor issues from the console. Exit the process with <kbd>Ctrl</kbd> + <kbd>C</kbd>
or by calling the shutdown hotkeys configured for your environment.
On Windows you can achieve the same by invoking `Run Whisper.bat --headless`.

### Configuration

## Documentation map
| File | Summary |
| --- | --- |
| `docs/first-run.md` | How the onboarding wizard captures only the minimal information (hotkey, storage root, curated model) before deferring advanced tweaks to the main UI. |
| `docs/deployment.md` | Strategies for relocating the directories defined in `advanced.storage.*` and keeping cold starts short across machines. |
| `docs/dependency-audit.md` | Running the dependency audit against `requirements.txt`, `requirements-extras.txt`, and friends to detect missing packages quickly. |
| `docs/model-loading-flow.md` | Timeline for the CT2 model lifecycle and how the state manager keeps the tray icon accurate during downloads and reloads. |
| `docs/ui_vars.md` | Mapping between CustomTkinter variables and configuration keys, updated to reflect the advanced namespace. |

### Window-only fallback mode

Environments that block access to the operating system tray (for example, missing `pystray`, restricted remote desktops, or
headless sessions) now trigger a controlled fallback. During startup the diagnostics report records the problem under the
**System Tray** check, and the runtime emits a structured warning with the `ui.tray_icon.unavailable` event. The application
automatically opens the settings window so you can continue managing recordings without relying on the tray icon. Install the
`pystray` and `Pillow` packages and ensure a compatible desktop session is available to restore full tray functionality.

### ASR backend policy

Whisper Flash Transcriber now distributes only the faster-whisper/CTranslate2 backend. The legacy Transformers
pipeline has been removed from the packaged application. If you depend on the Transformers stack (for example,
to experiment with custom attention implementations or torch-native quantization), fork the project, install
`requirements-legacy.txt`, and re-enable the removed backend modules.

### Custom installation directories

The application allows you to relocate heavyweight assets so that ephemeral or slow system drives do not become bottlenecks. The
following directories can be configured either directly in `config.json` or through the first-run wizard and the Settings UI:

- **`python_packages_dir`** — Target passed to `pip install --target` when optional packages (faster-whisper, ctranslate2,
  onnxruntime, etc.) are installed through the dependency remediation workflow. When you place this directory outside the active
virtual environment, ensure that `PYTHONPATH` includes the path before launching the application. The bootstrap logic adds the
directory to `sys.path`, but external scripts or shells may require explicit exports.
- **`vad_models_dir`** — Dedicated folder for the Silero VAD model. If empty, the packaged copy is copied into the directory on
  first use. Keep this path on a fast local drive to avoid I/O stalls during VAD activation.
- **`hf_cache_dir`** — Shared Hugging Face cache that backs `snapshot_download` calls used by the CTranslate2 runtime. The
  bootstrap sequence creates the directory and sets `HF_HOME`/`HUGGINGFACE_HUB_CACHE` accordingly.

Because all these directories default to the storage root, you can move the entire cache tree by changing `storage_root_dir` or
override each path individually for more granular layouts.

### Recording and Transcribing

- Press the configured hotkey to begin recording.
- Press again (or release, depending on the chosen mode) to stop.
- The application transcribes the captured audio and, if enabled, copies the final result to the clipboard and pastes it into the active window.

### Model download lifecycle

When a model install is required, the core service performs a deterministic sequence:

1. Persist `status=in_progress` with the selected model/backend so the UI and logs remain aligned.
2. Estimate download size via Hugging Face, add a 10% (minimum 256 MiB) safety margin, and abort with a descriptive error if free space is insufficient.
3. Clean up any stale directories, resume partial transfers when possible, and expose a cancellable progress bar.
4. Validate the installation (including essential files and metadata) before returning the application to `IDLE`.
5. Persist the resulting status (`success`, `skipped`, `cancelled`, or `error`) alongside the resolved installation path for future audits.

These safeguards ensure that repeated launches avoid redundant downloads and that any failure is actionable without inspecting the filesystem manually.

### Storage relocation workflow

Changing the storage root inside **Settings** immediately evaluates whether the previous base directory differs from the new target. When the user has not overridden the model cache or recordings directories, the configuration manager automatically moves the existing folders, skipping migration if the destination already hosts data. Every move operation is logged with structured metadata so audits can confirm where heavy assets live after the change.

### Advanced VAD controls

Voice Activity Detection now exposes pre- and post-speech padding in milliseconds. These values let you preserve audio context before the first spoken frame and keep trailing silence to avoid clipping. Invalid inputs automatically fall back to safe defaults, and the pipeline stays in sync with the stored configuration even when values are edited manually in `config.json`.

### Windows permissions and global hotkeys

The application registers global shortcuts using the [`keyboard`](https://github.com/boppreh/keyboard) library. Key suppression is intentionally disabled so that hotkeys work without elevated privileges on Windows. If you customize the code to block the underlying key events system-wide, make sure to run the application as an administrator to satisfy the library requirements.

## Testing and validation

Automated checks:

- `python -m compileall src` — validates that all Python modules compile.
- `pytest` — executes the available test suite.

Manual verification (recommended after storage or model changes):

1. **Model download resilience**
   - Delete or rename the cached model directory for a small model.
   - Launch the app and accept the download prompt.
   - Confirm that the download can be cancelled mid-transfer and that the settings window reflects the `cancelled` status before retrying.
   - Retry the download to ensure the installation completes and `status=success` is recorded.
2. **Storage relocation**
   - In Settings, change the storage root to an empty directory.
   - Ensure that both the model cache and recordings directories move automatically when no overrides are present.
   - Repeat the process with overrides enabled to confirm that migrations are skipped and logs report the decision.
3. **Dependency audit**
   - Run `pip list --outdated` inside the virtual environment to review available updates.
  - For each candidate library, cross-check release notes before upgrading and rerun `pytest` to verify compatibility.
  - Use `constraints.txt` during installations to remain within supported upper bounds, unless you are purposefully validating
    newer releases.

Document any deviations or failures inside the `plans/` folder so future operators can trace remediation steps.

## Architecture Overview

- **`main.py`:** Application entry point that initializes `AppCore` and the user interface.
- **`core.py`:** Coordinates modules and maintains global application state.
- **`ui_manager.py`:** Manages the GUI, settings window, and system tray icon.
- **`audio_handler.py`:** Captures audio from the microphone and routes it to storage.
- **`transcription_handler.py`:** Loads Whisper models and performs speech recognition.
- **`config_manager.py`:** Loads and persists user configuration.

For detailed developer notes and diagrams, review the files under the `docs/` directory.
