"""Interactive wizard displayed on the very first launch of the application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import customtkinter as ctk
from tkinter import filedialog, messagebox

from .. import config_manager as config_module
from ..logging_utils import StructuredMessage, get_logger
from ..model_manager import list_catalog, normalize_backend_label


LOGGER = get_logger("whisper_flash_transcriber.ui.first_run", component="FirstRunWizard")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_DIR = _REPO_ROOT / "plans"
_OPTIONAL_REQUIREMENTS = _REPO_ROOT / "requirements-optional.txt"


@dataclass(slots=True)
class FirstRunWizardResult:
    """Final payload returned by :class:`FirstRunWizard`."""

    config_updates: Dict[str, Any]
    selected_models: List[str]
    selected_packages: List[str]
    plan_path: Path | None


def _load_optional_packages() -> List[str]:
    """Parse ``requirements-optional.txt`` returning installable entries."""

    if not _OPTIONAL_REQUIREMENTS.exists():
        return []

    packages: List[str] = []
    try:
        with _OPTIONAL_REQUIREMENTS.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                packages.append(line)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning(
            "Failed to parse optional requirements: %s", exc, exc_info=True
        )
    return packages


class FirstRunWizard(ctk.CTkToplevel):
    """Multi-step assistant that captures the most relevant onboarding choices."""

    @classmethod
    def launch(
        cls, master, config_manager: "config_module.ConfigManager"
    ) -> FirstRunWizardResult | None:
        """Create, display, and block until the wizard terminates."""

        widget = cls(master, config_manager)
        master.wait_window(widget)
        return widget.result

    def __init__(
        self,
        master,
        config_manager: "config_module.ConfigManager",
    ) -> None:
        super().__init__(master)
        self._config_manager = config_manager
        self.result: FirstRunWizardResult | None = None

        self.withdraw()
        self.title("Welcome to Whisper Flash Transcriber")
        self.geometry("780x620")
        self.minsize(720, 560)
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._catalog = list_catalog()
        self._catalog.sort(key=lambda item: item.get("display_name") or item.get("id") or "")
        self._catalog_backend_map = {
            entry.get("id", ""): normalize_backend_label(entry.get("backend"))
            for entry in self._catalog
        }
        self._catalog_display_map = {
            entry.get("id", ""): entry.get("display_name") or entry.get("id", "")
            for entry in self._catalog
        }
        self._display_to_model_id = {
            display: model_id for model_id, display in self._catalog_display_map.items()
        }

        self._optional_packages = _load_optional_packages()

        self._state: Dict[str, Any] = self._build_initial_state()
        self._step_index = 0
        self._current_frame: ctk.CTkFrame | None = None

        self._directories_vars: Dict[str, ctk.StringVar] = {}
        self._asr_vars: Dict[str, Any] = {}
        self._preferences_vars: Dict[str, Any] = {}
        self._install_model_vars: Dict[str, ctk.BooleanVar] = {}
        self._install_package_vars: Dict[str, ctk.BooleanVar] = {}
        self._export_plan_var: ctk.BooleanVar | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._header_frame = ctk.CTkFrame(self)
        self._header_frame.grid(row=0, column=0, sticky="ew", padx=24, pady=(24, 12))
        self._header_frame.columnconfigure(0, weight=1)

        title_font = ctk.CTkFont(size=22, weight="bold")
        self._title_label = ctk.CTkLabel(self._header_frame, text="", font=title_font)
        self._title_label.grid(row=0, column=0, sticky="w")
        self._subtitle_label = ctk.CTkLabel(
            self._header_frame,
            text="",
            justify="left",
            wraplength=700,
        )
        self._subtitle_label.grid(row=1, column=0, sticky="w", pady=(8, 0))

        self._content_container = ctk.CTkFrame(self)
        self._content_container.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 12))
        self._content_container.rowconfigure(0, weight=1)
        self._content_container.columnconfigure(0, weight=1)

        self._nav_frame = ctk.CTkFrame(self)
        self._nav_frame.grid(row=2, column=0, sticky="ew", padx=24, pady=(12, 24))
        self._nav_frame.columnconfigure((0, 1, 2), weight=1)

        self._cancel_button = ctk.CTkButton(
            self._nav_frame,
            text="Cancel",
            command=self._on_cancel,
        )
        self._cancel_button.grid(row=0, column=0, sticky="w")

        self._back_button = ctk.CTkButton(
            self._nav_frame,
            text="Back",
            command=self._on_back,
        )
        self._back_button.grid(row=0, column=1, sticky="e", padx=(0, 12))

        self._next_button = ctk.CTkButton(
            self._nav_frame,
            text="Next",
            command=self._on_next,
        )
        self._next_button.grid(row=0, column=2, sticky="e")

        self._finish_button = ctk.CTkButton(
            self._nav_frame,
            text="Finish",
            command=self._on_finish,
        )
        self._finish_button.grid(row=0, column=2, sticky="e")
        self._finish_button.grid_remove()

        self._steps: List[dict[str, Any]] = [
            {
                "key": "directories",
                "title": "Choose your storage layout",
                "description": (
                    "Define where recordings and ASR assets will live. "
                    "Each directory can point to a dedicated drive."
                ),
                "builder": self._build_directories_step,
                "collector": self._collect_directories_step,
            },
            {
                "key": "asr",
                "title": "Pick the initial speech recognition stack",
                "description": (
                    "Select the curated backend and model that will power the initial "
                    "transcriptions. Choices can be adjusted later inside the settings window."
                ),
                "builder": self._build_asr_step,
                "collector": self._collect_asr_step,
            },
            {
                "key": "preferences",
                "title": "Fine-tune capture and paste behaviour",
                "description": (
                    "Configure voice activity detection and automatic paste preferences."
                ),
                "builder": self._build_preferences_step,
                "collector": self._collect_preferences_step,
            },
            {
                "key": "install",
                "title": "Prepare optional downloads",
                "description": (
                    "Queue curated models and optional Python packages for installation "
                    "after the wizard completes. Downloads will start once the main "
                    "application finishes bootstrapping."
                ),
                "builder": self._build_installation_step,
                "collector": self._collect_installation_step,
            },
            {
                "key": "summary",
                "title": "Review and confirm",
                "description": (
                    "A consolidated checklist of the onboarding decisions. You can also "
                    "export the selections to the \"plans\" folder as a reproducible snapshot."
                ),
                "builder": self._build_summary_step,
                "collector": self._collect_summary_step,
            },
        ]

        self.deiconify()
        self.after(50, self.focus_force)
        self._show_step(0)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _build_initial_state(self) -> Dict[str, Any]:
        cfg = self._config_manager.config
        return {
            "storage_root_dir": cfg.get(
                config_module.STORAGE_ROOT_DIR_CONFIG_KEY,
                config_module.DEFAULT_CONFIG[config_module.STORAGE_ROOT_DIR_CONFIG_KEY],
            ),
            "models_storage_dir": cfg.get(
                config_module.MODELS_STORAGE_DIR_CONFIG_KEY,
                cfg.get(config_module.STORAGE_ROOT_DIR_CONFIG_KEY),
            ),
            "recordings_dir": cfg.get(
                config_module.RECORDINGS_DIR_CONFIG_KEY,
                config_module.DEFAULT_CONFIG[config_module.RECORDINGS_DIR_CONFIG_KEY],
            ),
            "asr_backend": cfg.get(
                config_module.ASR_BACKEND_CONFIG_KEY,
                config_module.DEFAULT_CONFIG[config_module.ASR_BACKEND_CONFIG_KEY],
            ),
            "asr_model_id": cfg.get(
                config_module.ASR_MODEL_ID_CONFIG_KEY,
                config_module.DEFAULT_CONFIG[config_module.ASR_MODEL_ID_CONFIG_KEY],
            ),
            "use_vad": bool(cfg.get(config_module.USE_VAD_CONFIG_KEY, False)),
            "auto_paste": bool(cfg.get("auto_paste", True)),
            "agent_auto_paste": bool(cfg.get("agent_auto_paste", True)),
            "auto_paste_modifier": cfg.get("auto_paste_modifier", "auto"),
            "selected_models": [],
            "selected_packages": [],
            "export_plan": False,
        }

    # ------------------------------------------------------------------
    # Step rendering
    # ------------------------------------------------------------------
    def _show_step(self, index: int) -> None:
        self._step_index = index
        step = self._steps[self._step_index]

        self._title_label.configure(text=step["title"])
        self._subtitle_label.configure(text=step["description"])

        if self._current_frame is not None:
            self._current_frame.destroy()
            self._current_frame = None

        builder: Callable[[], ctk.CTkFrame] = step["builder"]
        self._current_frame = builder()

        self._back_button.configure(state="normal" if self._step_index > 0 else "disabled")

        if self._step_index == len(self._steps) - 1:
            self._next_button.grid_remove()
            self._finish_button.grid()
        else:
            self._finish_button.grid_remove()
            self._next_button.grid()
            self._next_button.configure(state="normal")

    def _build_directories_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_container)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        descriptions = {
            "storage_root_dir": "Base directory for caches and shared assets.",
            "models_storage_dir": "Overrides the storage location for ASR models.",
            "recordings_dir": "Destination folder for WAV recordings.",
        }

        self._directories_vars = {}
        for row, key in enumerate((
            "storage_root_dir",
            "models_storage_dir",
            "recordings_dir",
        )):
            label = ctk.CTkLabel(frame, text=descriptions[key])
            label.grid(row=row * 2, column=0, columnspan=3, sticky="w", pady=(10 if row else 0, 4))

            path_var = ctk.StringVar(value=str(self._state.get(key, "")))
            self._directories_vars[key] = path_var

            entry = ctk.CTkEntry(frame, textvariable=path_var)
            entry.grid(row=row * 2 + 1, column=0, columnspan=2, sticky="ew", padx=(0, 12))

            browse_button = ctk.CTkButton(
                frame,
                text="Browse",
                width=110,
                command=lambda var=path_var: self._pick_directory(var),
            )
            browse_button.grid(row=row * 2 + 1, column=2, sticky="e")

        return frame

    def _build_asr_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_container)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        backend_options = sorted(
            {normalize_backend_label(entry.get("backend")) or "ctranslate2" for entry in self._catalog}
        )
        if not backend_options:
            backend_options = [normalize_backend_label(self._state.get("asr_backend")) or "ctranslate2"]

        self._asr_vars = {}
        backend_var = ctk.StringVar(value=normalize_backend_label(self._state.get("asr_backend")))
        self._asr_vars["backend_var"] = backend_var

        backend_label = ctk.CTkLabel(
            frame,
            text="ASR backend",
            font=ctk.CTkFont(weight="bold"),
        )
        backend_label.grid(row=0, column=0, sticky="w", pady=(12, 6))
        backend_menu = ctk.CTkOptionMenu(
            frame,
            variable=backend_var,
            values=[option or "ctranslate2" for option in backend_options],
        )
        backend_menu.grid(row=1, column=0, sticky="ew")

        model_label = ctk.CTkLabel(
            frame,
            text="Curated model",
            font=ctk.CTkFont(weight="bold"),
        )
        model_label.grid(row=2, column=0, sticky="w", pady=(24, 6))

        model_values = [self._catalog_display_map.get(item.get("id", ""), item.get("id", "")) for item in self._catalog]
        if not model_values:
            model_values = [self._state.get("asr_model_id", "") or ""]

        initial_model_display = self._catalog_display_map.get(self._state.get("asr_model_id"), model_values[0])
        model_var = ctk.StringVar(value=initial_model_display)
        self._asr_vars["model_var"] = model_var

        model_menu = ctk.CTkOptionMenu(
            frame,
            variable=model_var,
            values=model_values,
            command=self._on_model_changed,
        )
        model_menu.grid(row=3, column=0, sticky="ew")

        return frame

    def _build_preferences_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_container)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        self._preferences_vars = {}
        vad_var = ctk.BooleanVar(value=bool(self._state.get("use_vad", False)))
        auto_paste_var = ctk.BooleanVar(value=bool(self._state.get("auto_paste", True)))
        agent_paste_var = ctk.BooleanVar(value=bool(self._state.get("agent_auto_paste", True)))

        self._preferences_vars["use_vad"] = vad_var
        self._preferences_vars["auto_paste"] = auto_paste_var
        self._preferences_vars["agent_auto_paste"] = agent_paste_var

        vad_switch = ctk.CTkSwitch(
            frame,
            text="Enable voice activity detection (silence trimming)",
            variable=vad_var,
        )
        vad_switch.grid(row=0, column=0, sticky="w", pady=(24, 12))

        auto_paste_switch = ctk.CTkSwitch(
            frame,
            text="Automatically paste transcriptions after completion",
            variable=auto_paste_var,
        )
        auto_paste_switch.grid(row=1, column=0, sticky="w", pady=(12, 12))

        agent_paste_switch = ctk.CTkSwitch(
            frame,
            text="Apply the same behaviour when Agent Mode is enabled",
            variable=agent_paste_var,
        )
        agent_paste_switch.grid(row=2, column=0, sticky="w", pady=(12, 12))

        modifier_label = ctk.CTkLabel(
            frame,
            text="Automatic paste modifier",
            font=ctk.CTkFont(weight="bold"),
        )
        modifier_label.grid(row=3, column=0, sticky="w", pady=(18, 6))

        modifier_options = ["auto", "ctrl+v", "shift+insert", "enter"]
        current_modifier = self._state.get("auto_paste_modifier", "auto")
        if current_modifier not in modifier_options:
            modifier_options.append(current_modifier)

        modifier_var = ctk.StringVar(value=current_modifier)
        self._preferences_vars["auto_paste_modifier"] = modifier_var

        modifier_menu = ctk.CTkOptionMenu(
            frame,
            variable=modifier_var,
            values=modifier_options,
        )
        modifier_menu.grid(row=4, column=0, sticky="w")

        return frame

    def _build_installation_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_container)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(frame)
        scroll.grid(row=0, column=0, sticky="nsew", pady=12)
        scroll.columnconfigure(0, weight=1)

        models_header = ctk.CTkLabel(
            scroll,
            text="Curated ASR models",
            font=ctk.CTkFont(weight="bold"),
        )
        models_header.grid(row=0, column=0, sticky="w", pady=(12, 6))

        self._install_model_vars = {}
        if self._catalog:
            for idx, entry in enumerate(self._catalog, start=1):
                model_id = entry.get("id", "")
                display = self._catalog_display_map.get(model_id, model_id)
                var = ctk.BooleanVar(value=model_id == self._state.get("asr_model_id"))
                checkbox = ctk.CTkCheckBox(
                    scroll,
                    text=f"{display} ({model_id})",
                    variable=var,
                )
                checkbox.grid(row=idx, column=0, sticky="w", pady=(2, 2))
                self._install_model_vars[model_id] = var
        else:
            placeholder = ctk.CTkLabel(
                scroll,
                text="No curated models detected in the bundled catalog.",
            )
            placeholder.grid(row=1, column=0, sticky="w")

        packages_header = ctk.CTkLabel(
            scroll,
            text="Optional Python packages",
            font=ctk.CTkFont(weight="bold"),
        )
        packages_header.grid(row=len(self._install_model_vars) + 2, column=0, sticky="w", pady=(24, 6))

        self._install_package_vars = {}
        if self._optional_packages:
            base_row = len(self._install_model_vars) + 3
            for offset, package in enumerate(self._optional_packages):
                var = ctk.BooleanVar(value=False)
                checkbox = ctk.CTkCheckBox(scroll, text=package, variable=var)
                checkbox.grid(row=base_row + offset, column=0, sticky="w", pady=(2, 2))
                self._install_package_vars[package] = var
        else:
            placeholder = ctk.CTkLabel(
                scroll,
                text="No optional dependencies were found in requirements-optional.txt.",
            )
            placeholder.grid(row=len(self._install_model_vars) + 3, column=0, sticky="w")

        return frame

    def _build_summary_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_container)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        summary = ctk.CTkScrollableFrame(frame)
        summary.grid(row=0, column=0, sticky="nsew", pady=(12, 6))
        summary.columnconfigure(0, weight=1)

        sections = [
            (
                "Directories",
                [
                    f"Storage root → {self._state['storage_root_dir']}",
                    f"Models directory → {self._state['models_storage_dir']}",
                    f"Recordings directory → {self._state['recordings_dir']}",
                ],
            ),
            (
                "Speech recognition",
                [
                    f"Backend → {normalize_backend_label(self._state['asr_backend'])}",
                    f"Model → {self._state['asr_model_id']}",
                ],
            ),
            (
                "Behaviour",
                [
                    "Voice activity detection → "
                    + ("Enabled" if self._state.get("use_vad") else "Disabled"),
                    "Automatic paste → "
                    + ("Enabled" if self._state.get("auto_paste") else "Disabled"),
                    "Agent mode paste → "
                    + ("Enabled" if self._state.get("agent_auto_paste") else "Disabled"),
                    f"Paste modifier → {self._state.get('auto_paste_modifier')}",
                ],
            ),
            (
                "Installation queue",
                [
                    "Models → "
                    + (", ".join(self._state.get("selected_models") or ["None"]) or "None"),
                    "Packages → "
                    + (", ".join(self._state.get("selected_packages") or ["None"]) or "None"),
                ],
            ),
        ]

        row = 0
        for title, items in sections:
            header = ctk.CTkLabel(summary, text=title, font=ctk.CTkFont(weight="bold"))
            header.grid(row=row, column=0, sticky="w", pady=(12 if row else 0, 6))
            row += 1
            for item in items:
                var = ctk.BooleanVar(value=True)
                checkbox = ctk.CTkCheckBox(summary, text=item, variable=var)
                checkbox.configure(state="disabled")
                checkbox.grid(row=row, column=0, sticky="w", pady=(2, 2))
                row += 1

        self._export_plan_var = ctk.BooleanVar(value=bool(self._state.get("export_plan", False)))
        export_checkbox = ctk.CTkCheckBox(
            frame,
            text="Export snapshot to the plans/ directory",
            variable=self._export_plan_var,
        )
        export_checkbox.grid(row=1, column=0, sticky="w", pady=(12, 0))

        return frame

    # ------------------------------------------------------------------
    # Collectors
    # ------------------------------------------------------------------
    def _collect_directories_step(self) -> bool:
        collected: Dict[str, str] = {}
        for key, var in self._directories_vars.items():
            value = var.get().strip()
            if not value:
                messagebox.showerror(
                    "Missing directory",
                    "Please select a valid directory before continuing.",
                    parent=self,
                )
                return False
            try:
                expanded = Path(value).expanduser()
            except Exception:
                messagebox.showerror(
                    "Invalid directory",
                    f"The path '{value}' could not be resolved.",
                    parent=self,
                )
                return False
            collected[key] = str(expanded)

        self._state.update(collected)
        return True

    def _collect_asr_step(self) -> bool:
        backend_var: ctk.StringVar = self._asr_vars.get("backend_var")
        model_var: ctk.StringVar = self._asr_vars.get("model_var")

        backend_value = normalize_backend_label(backend_var.get()) if backend_var else "ctranslate2"
        model_display = model_var.get().strip() if model_var else ""
        model_id = self._display_to_model_id.get(model_display, model_display)
        if not model_id:
            messagebox.showerror(
                "Model required",
                "Select one of the curated models before continuing.",
                parent=self,
            )
            return False

        curated_backend = self._catalog_backend_map.get(model_id)
        if curated_backend:
            backend_value = curated_backend
            backend_var.set(curated_backend)

        self._state.update(
            {
                "asr_backend": backend_value,
                "asr_model_id": model_id,
            }
        )
        return True

    def _collect_preferences_step(self) -> bool:
        modifier_var: ctk.StringVar = self._preferences_vars.get("auto_paste_modifier")
        modifier_value = (modifier_var.get() if modifier_var else "auto").strip() or "auto"

        self._state.update(
            {
                "use_vad": bool(self._preferences_vars["use_vad"].get()),
                "auto_paste": bool(self._preferences_vars["auto_paste"].get()),
                "agent_auto_paste": bool(self._preferences_vars["agent_auto_paste"].get()),
                "auto_paste_modifier": modifier_value,
            }
        )
        return True

    def _collect_installation_step(self) -> bool:
        selected_models = [
            model_id for model_id, var in self._install_model_vars.items() if bool(var.get())
        ]
        selected_packages = [
            package for package, var in self._install_package_vars.items() if bool(var.get())
        ]

        self._state.update(
            {
                "selected_models": selected_models,
                "selected_packages": selected_packages,
            }
        )
        return True

    def _collect_summary_step(self) -> bool:
        if self._export_plan_var is not None:
            self._state["export_plan"] = bool(self._export_plan_var.get())
        return True

    # ------------------------------------------------------------------
    # Navigation callbacks
    # ------------------------------------------------------------------
    def _on_back(self) -> None:
        if self._step_index == 0:
            return
        self._show_step(self._step_index - 1)

    def _on_next(self) -> None:
        if not self._collect_current_step():
            return
        self._show_step(self._step_index + 1)

    def _on_finish(self) -> None:
        if not self._collect_current_step():
            return
        self._finalize()

    def _on_cancel(self) -> None:
        LOGGER.info(
            StructuredMessage(
                "First run wizard cancelled by the user.",
                event="first_run.cancelled",
            )
        )
        self.result = None
        self.grab_release()
        self.destroy()

    def _collect_current_step(self) -> bool:
        collector: Callable[[], bool] = self._steps[self._step_index]["collector"]
        return collector()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def _finalize(self) -> None:
        config_updates = {
            config_module.STORAGE_ROOT_DIR_CONFIG_KEY: self._state["storage_root_dir"],
            config_module.MODELS_STORAGE_DIR_CONFIG_KEY: self._state["models_storage_dir"],
            config_module.RECORDINGS_DIR_CONFIG_KEY: self._state["recordings_dir"],
            config_module.ASR_BACKEND_CONFIG_KEY: self._state["asr_backend"],
            config_module.ASR_MODEL_ID_CONFIG_KEY: self._state["asr_model_id"],
            config_module.USE_VAD_CONFIG_KEY: bool(self._state.get("use_vad", False)),
            "auto_paste": bool(self._state.get("auto_paste", True)),
            "agent_auto_paste": bool(self._state.get("agent_auto_paste", True)),
            "auto_paste_modifier": self._state.get("auto_paste_modifier", "auto"),
            config_module.FIRST_RUN_COMPLETED_CONFIG_KEY: True,
        }

        plan_path = None
        if self._state.get("export_plan"):
            plan_path = self._write_plan_snapshot()

        self.result = FirstRunWizardResult(
            config_updates=config_updates,
            selected_models=list(self._state.get("selected_models", [])),
            selected_packages=list(self._state.get("selected_packages", [])),
            plan_path=plan_path,
        )

        LOGGER.info(
            StructuredMessage(
                "First run wizard completed.",
                event="first_run.completed",
                details={
                    "models": self.result.selected_models,
                    "packages": self.result.selected_packages,
                    "plan": str(plan_path) if plan_path else None,
                },
            )
        )
        self.grab_release()
        self.destroy()

    def _write_plan_snapshot(self) -> Path | None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        file_path = _PLANS_DIR / f"first-run-{timestamp}.md"
        try:
            _PLANS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error(
                "Unable to create plans directory '%s': %s", _PLANS_DIR, exc, exc_info=True
            )
            return None

        content_lines = [
            f"# First Run Wizard Snapshot - {timestamp} UTC",
            "",
            "## Directories",
            f"- [x] Storage root: `{self._state['storage_root_dir']}`",
            f"- [x] Models directory: `{self._state['models_storage_dir']}`",
            f"- [x] Recordings directory: `{self._state['recordings_dir']}`",
            "",
            "## Speech recognition",
            f"- [x] Backend: `{normalize_backend_label(self._state['asr_backend'])}`",
            f"- [x] Model: `{self._state['asr_model_id']}`",
            "",
            "## Behaviour",
            f"- [x] Voice activity detection: {'Enabled' if self._state.get('use_vad') else 'Disabled'}",
            f"- [x] Automatic paste: {'Enabled' if self._state.get('auto_paste') else 'Disabled'}",
            f"- [x] Agent mode paste: {'Enabled' if self._state.get('agent_auto_paste') else 'Disabled'}",
            f"- [x] Paste modifier: `{self._state.get('auto_paste_modifier')}`",
            "",
            "## Installation queue",
            "- [x] Models:",
        ]

        models = self._state.get("selected_models") or []
        if models:
            content_lines.extend([f"  - `{model}`" for model in models])
        else:
            content_lines.append("  - _None selected_")

        content_lines.append("- [x] Packages:")
        packages = self._state.get("selected_packages") or []
        if packages:
            content_lines.extend([f"  - `{package}`" for package in packages])
        else:
            content_lines.append("  - _None selected_")

        content_lines.append("")
        content_lines.append("Generated automatically by the onboarding assistant.")

        try:
            file_path.write_text("\n".join(content_lines), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error(
                "Unable to persist first run plan snapshot '%s': %s",
                file_path,
                exc,
                exc_info=True,
            )
            return None

        return file_path

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _pick_directory(self, target_var: ctk.StringVar) -> None:
        selected = filedialog.askdirectory(parent=self, mustexist=False)
        if selected:
            target_var.set(selected)

    def _on_model_changed(self, display_value: str) -> None:
        model_id = self._display_to_model_id.get(display_value, display_value)
        curated_backend = self._catalog_backend_map.get(model_id)
        if curated_backend and "backend_var" in self._asr_vars:
            backend_var: ctk.StringVar = self._asr_vars["backend_var"]
            backend_var.set(curated_backend)
