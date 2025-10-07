# Dependency audit workflow

The dependency audit panel compares the active Python environment with the manifests shipped with the project. It is designed to flag missing or outdated packages minutes after bootstrap, before the minimal workflow is affected.

## When the audit runs
1. **AppCore bootstrap** – Right after loading the configuration, `AppCore` calls `audit_environment()` and records the result via `ConfigManager.record_runtime_notice()`.
2. **State event** – The core emits `StateEvent.DEPENDENCY_AUDIT_READY` with a textual summary as soon as `StateManager` is ready.
3. **UI delivery** – Once `UIManager` subscribes, the non-modal panel opens automatically and remains accessible until the operator closes it.

## Manifest coverage
`src/utils/dependency_audit.py` scans the following files:

- `requirements.txt` – mandatory dependencies for the hotkey → capture → CTranslate2 → paste loop.
- `requirements-extras.txt` – opt-in packages (Gemini/OpenRouter, Playwright, ONNX Runtime, accelerate/datasets).
- `requirements-optional.txt` – GPU-centric add-ons such as `bitsandbytes`.
- `requirements-test.txt` – development tooling.

The panel shows which manifest introduced each requirement so you can decide whether the missing package is truly required for your deployment.

## Diagnostic categories
| Category | Meaning | Suggested action |
| --- | --- | --- |
| Missing dependency | Package not found in the current environment. | Copy the generated `pip install ...` command and run it. |
| Version mismatch | Installed version falls outside the specifier listed in the manifest. | Use the copy button to upgrade/downgrade with `pip install --upgrade ...`. |
| Hash mismatch | The installed wheel hash diverges from the recorded hash (when available). | Reinstall with `pip install --force-reinstall --no-cache-dir ...` or align the manifest hash. |

> **Note:** When the manifest does not include hashes the last column remains empty.

## Using the panel
1. **Summary** – The header displays the global status (for example `Dependency audit completed — 1 missing`) and the UTC timestamp converted to the local timezone.
2. **Per-category lists** – Each card enumerates the requirement, the origin file/line, the installed version (if any), and available hashes.
3. **Individual commands** – “Copy command” produces a `python -m pip install ...` command that already includes extras and specifiers.
4. **Copy all** – Aggregates deduplicated commands so they can be pasted into a shell and executed sequentially.
5. **Open documentation** – Opens this file in the default browser for quick reference.

## Common issues
| Symptom | Likely cause | Mitigation |
| --- | --- | --- |
| Panel reports “Dependency audit failed” | `packaging` / `importlib.metadata` missing or a manifest is unreadable. | Activate the virtual environment and reinstall with `pip install -r requirements.txt`. |
| Panel is empty but the environment is still broken | The dependency in question lives outside the manifests or is filtered by a platform marker. | Review the manifests, add the missing dependency, and rerun the audit. |
| Hash mismatches keep returning | Wheels come from a private mirror and expose different hashes in `direct_url.json`. | Update the manifest hash or reinstall with `pip install --no-deps --force-reinstall <wheel>`. |
| Copy buttons do nothing | Clipboard blocked by the OS or the Tk window lacks focus. | Give the main window focus or run the application with the required privileges. |

## Suggested follow-ups
1. Run the audit inside CI to validate pull requests that touch any `requirements*.txt` file.
2. Persist the latest report under `logs/` for historical analysis.
3. Expand the parser to support nested manifests (`-r path/to/requirements.txt`) if the repository grows additional components.
