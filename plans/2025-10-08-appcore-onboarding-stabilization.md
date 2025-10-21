# AppCore UI Manager Integration and Onboarding Stabilization Plan (08 Oct 2025)

## Context
This document maps the existing `AppCore.ui_manager` setter workflow and delineates the instrumentation and manual validation actions required to stabilize the onboarding sequence. References are based on `src/core.py` (lines 329-489) and associated onboarding components.

## Current `AppCore.ui_manager` Setter Flow
1. **State subscription and dependency surface**
   - When `AppCore.ui_manager` receives an instance, it persists it in `_ui_manager`, subscribes `ui_manager.update_tray_icon` to the `StateManager`, and publishes dependency audit information immediately to keep the tray icon state synchronized.
   - Expected log events: dependency audit replay via `_publish_dependency_audit_state_event()` and `_present_dependency_audit_to_ui()` (look for `dependency_audit` prefixes).
2. **Hotkey manager bootstrap**
   - The setter resolves `hotkey_config_path`, instantiates `KeyboardHotkeyManager`, and creates synchronization primitives (`stop_reregister_event`, `stop_health_check_event`) for background stability threads.
   - Auxiliary threads are registered with semantic labels (`PeriodicHotkeyReregister`, `HotkeyHealthThread`) to support tracing in logs originating from `KeyboardHotkeyManager`.
3. **Initial configuration sync**
   - `_apply_initial_config_to_core_attributes()` runs immediately after the hotkey manager setup, ensuring runtime attributes such as compute-type, record mode, and VAD padding mirror persisted configuration prior to event dispatch.
   - Blocking conditions: any `ConfigPersistenceError` raised during config access; verify for warnings tagged `ConfigManager` in the log stream.
4. **Onboarding gate**
   - The setter resolves `model_download_timeout` and invokes `_maybe_run_initial_onboarding()` when `_onboarding_enabled` is true. Headless sessions skip this step with a debug log `Initial onboarding disabled for this session; skipping wizard launch.`
   - `_maybe_run_initial_onboarding()` queries `ConfigManager.describe_persistence_state()`; failure to inspect persistence aborts the wizard with a warning log and no further action.
   - If the snapshot reports `first_run=True`, `_launch_onboarding(..., force=True, reason="startup")` executes and stores the resulting outcome in `_last_onboarding_outcome`.

## Onboarding Flow Stabilization
```
main.py bootstrap
  -> configure_environment / diagnostics (logging: startup.*)
  -> AppCore(...)  # sets enable_onboarding based on --headless
  -> UIManager(...) and AppCore.ui_manager = instance
       • subscriptions + hotkey manager init (logs: dependency_audit.*, hotkey.*)
       • _apply_initial_config_to_core_attributes()
       • _maybe_run_initial_onboarding()
            -> describe_persistence_state()  (logs: onboarding.snapshot | warnings on failure)
            -> _launch_onboarding(force=True, reason="startup")
                 -> _run_onboarding_wizard()
                      · model_manager.list_catalog() (MODEL logger diagnostics)
                      · FirstRunWizard(...) and wizard.run()
                           ↳ UI thread must remain responsive (log: "Launching onboarding wizard for first run.")
                 -> _apply_onboarding_result() after wizard.run() returns
```
- **Blocking points**
  - `describe_persistence_state()` raising, logged as `Unable to inspect persistence state for onboarding` and short-circuiting the flow.
  - `_run_onboarding_wizard()` raising or returning `None`, which logs `Onboarding wizard dismissed without changes`.
  - UI deadlock if `wizard.run()` is invoked before Tk is fully initialized; watch for missing `ui.mainloop.start` in logs.
- **Expected log sequence**
  - `Launching onboarding wizard for first run.` (INFO)
  - `Onboarding wizard dismissed without changes (...)` or subsequent `Applying onboarding configuration updates (...)`.
  - If download is triggered, follow-up logs from `FirstRunWizard` handlers (prefixed by `onboarding.download`).

## Stabilization Checklist
1. **Instrumentation expansion**
   - [ ] Add structured debug logs immediately before and after `wizard.run()` inside `_run_onboarding_wizard()` to capture timing and Tk focus metadata.
   - [ ] Capture `self.main_tk_root.focus_get()` in the pre-run log to diagnose focus loss.
   - [ ] Include wizard outcome summary (`keys` count, download request flags) after `_apply_onboarding_result()`.
2. **Windows 11 focus validation**
   - [ ] Launch the application on Windows 11 and confirm `FirstRunWizard` gains focus.
   - [ ] Observe logs for the new `wizard.focus_acquired` marker; if absent, record the active window via `GetForegroundWindow()` instrumentation.
   - [ ] Command: `python src/main.py` (ensure not using `--headless`). Files to inspect: `src/onboarding/first_run_wizard.py`, `src/core.py`.
3. **Headless fallback behaviour**
   - [ ] Run `python src/main.py --headless` to ensure `_maybe_run_initial_onboarding()` emits the skip debug log and no Tk windows are created.
   - [ ] Verify `HeadlessEventLoop` continues to handle timers; capture logs tagged `headless.messagebox` if dialogs are invoked.
   - [ ] Document any residual attempts to create `FirstRunWizard` when headless (should be none).
4. **Log review pipeline**
   - [ ] Tail `logs/whisper_flash_transcriber.log` during wizard execution to verify new markers around `wizard.run()` and onboarding outcomes.
   - [ ] Use `rg "wizard" logs/` to locate focus diagnostics quickly.
5. **Configuration persistence audit**
   - [ ] After completing the wizard, confirm `ConfigManager.save_config()` persisted `first_run=False` and any hotkey overrides.
   - [ ] Diff `config.json` and ensure new compute-type values appear in `_apply_initial_config_to_core_attributes()` debug statements.

## Responsibilities and Manual Validation Matrix
| Area | Owner | Validation Steps | Notes |
| --- | --- | --- | --- |
| Logging instrumentation | Core engineering | 1) Implement pre/post `wizard.run()` logs.<br>2) Simulate success and cancel flows to verify markers. | Do **not** introduce new automated test suites; rely on manual log inspection. |
| UI focus handling | Desktop QA | 1) Execute a short (10s) recording after onboarding completes.<br>2) Confirm tray icon state updates and focus returns to settings window. | Record a short screen capture as evidence. |
| Wizard cancellation | QA + UX | 1) Launch wizard and cancel during the first step.<br>2) Confirm log entry `Onboarding wizard dismissed without changes` and no config mutations. | Document timestamped log excerpt. |
| Exception handling simulation | Core engineering | 1) Temporarily patch `FirstRunWizard.run` to raise `RuntimeError` (locally) and observe recovery path.<br>2) Ensure `_onboarding_active` flag resets and application remains operable. | Remove the patch after validation; only manual experiment, no committed test code. |
| Headless behaviour | DevOps | 1) Run `python src/main.py --headless` on CI-like environment.<br>2) Check logs for `Initial onboarding disabled` message and absence of Tk window warnings. | Confirm CLI exit via `Ctrl+C`; no GUI dependencies should initialize. |

_No new automated test suites should be created for this stabilization effort; validations are manual, instrumentation-driven, and log-audited._
