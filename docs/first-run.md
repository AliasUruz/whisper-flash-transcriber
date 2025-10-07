# First Run Wizard

The first-launch experience introduces a dedicated wizard that captures the
minimum configuration required to operate Whisper Flash Transcriber. The flow is
executed before the tray UI is created and runs once unless the configuration is
reset.

## Lifecycle

1. `main.py` instantiates `ConfigManager` and persists default files.
2. If `config_manager.is_first_run()` returns `True`, a hidden Tk root window is
   created and passed to `FirstRunWizard.launch`.
3. The wizard gathers the user choices, persists them through
   `ConfigManager.apply_updates`, and optionally exports a Markdown snapshot to
   `plans/`.
4. The application continues with `AppCore` and `UIManager` initialization. When
   models were selected for immediate installation the bootstrapper spawns a
   background thread that calls `model_manager.ensure_download` for each model.

Cancelling the wizard aborts the startup sequence so the user can relaunch it
later without a partially configured profile.

## Step breakdown

| Step | Description |
| --- | --- |
| Directories | Choose the storage root, ASR models directory, and recordings folder. Each path supports browsing via native file dialogs. |
| ASR bootstrap | Select the curated backend and model. The backend automatically follows the curated catalog entry for the selected model. |
| Capture preferences | Toggle voice activity detection and automatic paste behaviour, including agent-mode overrides and the modifier strategy. |
| Optional installations | Mark curated models for immediate download and optional Python packages for manual installation. |
| Summary | Review the selections through read-only checklists and choose whether to export a Markdown plan to `plans/`. |

## Exported plan format

When the user requests an export the wizard creates
`plans/first-run-<timestamp>.md`. The document is fully self-contained and
includes:

- The resolved directories.
- The ASR backend and model pair.
- Voice activity detection and paste preferences.
- The list of models queued for installation.
- The list of Python packages flagged for manual setup.

This artifact is intended for reproducibility audits and can be committed to the
repository if the operator wants to track environment provenance.
