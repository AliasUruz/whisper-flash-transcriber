# First-run Wizard

The onboarding wizard exists to collect just enough information to run the hotkey → capture → CTranslate2 → paste loop. Everything else belongs to the advanced namespace and stays hidden until the full settings window is available.

## Lifecycle
1. `main.py` provisions the profile directory and instantiates `ConfigManager`.
2. When `config_manager.is_first_run()` returns `True`, a hidden Tk root window launches `FirstRunWizard.launch()` before the tray UI appears.
3. The wizard gathers the minimal payload, calls `ConfigManager.apply_updates()` with the resulting dictionary, and optionally exports a Markdown plan under `plans/`.
4. Control returns to `AppCore` and `UIManager`. If the selected Whisper model is not cached, the bootstrap thread triggers an immediate download so the first transcription runs without further prompts.

Cancelling the wizard aborts the bootstrap. You can relaunch the application later without leaving behind a partially configured profile.

## Data collected by the wizard
| Step | Fields | Notes |
| --- | --- | --- |
| Hotkey | `record_key`, `record_mode`, `auto_paste` | Establishes how the minimal workflow is triggered and whether the transcript is pasted automatically. |
| Storage root | `advanced.storage.storage_root_dir` (plus derived directories when they follow the defaults) | The wizard only asks for a single root. Advanced overrides such as `models_storage_dir` or `recordings_dir` remain available from the full settings window. |
| Model selection | `asr_model_id` / `asr_backend` | The curated catalog is filtered based on the detected hardware profile. The backend mirrors the catalog entry, so users start with a valid CTranslate2 pairing. |
| Optional downloads | List of curated models | Allows preparing multiple models up front; all installs run sequentially after the wizard exits. |

Voice activity detection, AI corrections, directory overrides, and other advanced controls are deliberately skipped here. They remain accessible from Settings → Advanced once the minimal workflow is working.

## Exported plan format
When the operator asks for an export, the wizard writes `plans/first-run-<timestamp>.md` with a snapshot of:

- The hotkey strategy and auto-paste preference.
- The resolved storage root and any derived directories that follow it.
- The selected ASR backend/model pair and the download queue (if any).
- Environment hints such as `PYTHONPATH` additions when `advanced.storage.python_packages_dir` lives outside the virtual environment.

The plan is fully self-contained so you can check it into version control or share it across machines to reproduce the same minimal setup.
