# Dependency Bootstrapping Alignment Plan (06/10/2025)

## Dependency assistant rationalization
- **Instantiation map**
  - `src/main.py` (lines 619-631): constructs `DependencyInstaller` right after `ConfigManager`, injects the custom `python_packages_dir`, calls `ensure_in_sys_path()`, and streams advisory notes into the structured logger. No other module instantiates the helper during startup.
- **Runtime surface area exposed by `DependencyInstaller`** (`src/installers/dependency_installer.py`, lines 15-93)
  - `environment_notes`: read-only property exposing startup advisories after `ensure_in_sys_path()` refreshes the list.
  - `running_in_virtualenv`: property used to annotate environment notes when the interpreter is not isolated.
  - `ensure_in_sys_path()`: inserts the configured target directory into `sys.path` and recalculates the advisory list.
  - `build_pip_command(packages, upgrade=False)`: helper that composes the `pip` invocation pinned to the custom target directory.
  - `install(packages, upgrade=False, progress_callback=None, env=None)`: executes `pip install --target`, streams progress lines to optional callbacks, extends `PYTHONPATH` for the subprocess, and raises `RuntimeError` on non-zero exit codes.

## Advanced wizard reachability audit
- The advanced first-run wizard with asynchronous download queue (`src/ui/first_run_wizard.py`, lines 169-259) remains dormant because the active bootstrap path in `AppCore.bootstrap()` (`src/core.py`, lines 60-74 and subsequent import graph) never calls into this UI. The legacy onboarding entry point still imports from `src/onboarding/first_run_wizard`, evidencing the fork.
- Alignment options to resolve the dead path:
  - **Integrate the new wizard**: point the bootstrapper to the advanced UI, wire its completion payload into `ConfigManager`, and trigger download queue hand-off to the model controller.
    - Validate how the wizard exports plan snapshots to `plans/` and ensure permissions are satisfied.
  - **Remove the dormant implementation**: delete the advanced wizard module and associated assets if product requirements confirm it will not ship, reducing maintenance footprint.
  - **Reconcile both paths**: factor shared collectors/builders, host the UI under a single namespace, and guarantee that the wizard selection logic is explicit (feature flag, CLI switch, or config).

## Decision checklist (execute sequentially)
- [ ] Confirm UX/product requirements for first-run onboarding (stakeholders + latest design artifacts).
- [ ] Decide the fate of duplicated wizard code (`src/onboarding/first_run_wizard.py` vs `src/ui/first_run_wizard.py`) and document the rationale.
- [ ] Assess documentation deltas (`README.md`, `docs/`) required by the selected onboarding path and prepare updates.
- [ ] Enumerate and schedule manual regression checks post-refactor (e.g., startup diagnostics, dependency downloads, settings persistence, wizard completion export).

## Simplification recommendations and risk log
- **Recommended simplifications**
  - Remove `DependencyInstaller` entirely if no consumer outside `main.py` remains after onboarding alignment, replacing it with direct `sys.path` and `pip` handling or consolidating into `ConfigManager` utilities.
  - Collapse duplicated first-run wizard modules into a single implementation governed by feature flags instead of parallel directories.
- **Known risks**
  - Eliminating `DependencyInstaller` without replicating `environment_notes` could reduce operator visibility into PYTHONPATH requirements; ensure equivalent logging lives elsewhere.
  - Switching bootstrap flows may break assumptions in automated startup diagnostics (`src/startup_diagnostics.py`) if they expect the legacy wizard to populate certain config keys.
  - Integrating the download queue demands back-pressure coordination with `ModelDownloadController`; regression risk for existing auto-download logic must be mitigated through manual queue exhaustion tests.
