"""Utility class for installing Python packages into an isolated directory."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

_LOGGER = logging.getLogger(__name__)


class DependencyInstaller:
    """Install Python dependencies into a configurable target directory."""

    def __init__(
        self,
        python_packages_dir: str | os.PathLike[str],
        *,
        pip_executable: Sequence[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or _LOGGER
        self.python_packages_dir = Path(python_packages_dir).expanduser()
        self.python_packages_dir.mkdir(parents=True, exist_ok=True)
        self.pip_command = list(pip_executable) if pip_executable is not None else [sys.executable, "-m", "pip"]
        self._environment_notes: list[str] = []
        self._running_in_virtualenv = self._detect_virtualenv()
        self._refresh_environment_notes()

    @property
    def environment_notes(self) -> Sequence[str]:
        """Return advisory notes collected during environment inspection."""

        return tuple(self._environment_notes)

    @property
    def running_in_virtualenv(self) -> bool:
        """Return ``True`` if the current interpreter is inside a virtual environment."""

        return self._running_in_virtualenv

    def _detect_virtualenv(self) -> bool:
        base_prefix = getattr(sys, "base_prefix", sys.prefix)
        real_prefix = getattr(sys, "real_prefix", base_prefix)
        prefix = Path(sys.prefix)
        return prefix != Path(real_prefix) or prefix != Path(base_prefix)

    def _refresh_environment_notes(self) -> None:
        self._environment_notes.clear()
        target = str(self.python_packages_dir)
        if target not in sys.path:
            self._environment_notes.append(
                "Add the custom python_packages_dir to PYTHONPATH before launching the application."
            )
        if not self.running_in_virtualenv:
            self._environment_notes.append(
                "Running outside of a virtualenv; packages installed to the custom target will not shadow system packages without PYTHONPATH."  # noqa: E501
            )

    def ensure_in_sys_path(self) -> bool:
        """Insert the target directory into ``sys.path`` when missing."""

        target = str(self.python_packages_dir)
        if target in sys.path:
            return False
        sys.path.insert(0, target)
        self.logger.debug("Inserted '%s' into sys.path", target)
        self._refresh_environment_notes()
        return True

    def build_pip_command(self, packages: Iterable[str], *, upgrade: bool = False) -> list[str]:
        command = [*self.pip_command, "install", "--target", str(self.python_packages_dir)]
        if upgrade:
            command.append("--upgrade")
        command.extend(packages)
        return command

    def install(
        self,
        packages: Iterable[str],
        *,
        upgrade: bool = False,
        progress_callback: Callable[[str], None] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Install ``packages`` into the configured target directory."""

        packages = [str(pkg).strip() for pkg in packages if str(pkg).strip()]
        if not packages:
            self.logger.info("DependencyInstaller: no packages requested; skipping pip invocation.")
            return

        command = self.build_pip_command(packages, upgrade=upgrade)
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        existing_pythonpath = merged_env.get("PYTHONPATH")
        if existing_pythonpath:
            merged_env["PYTHONPATH"] = os.pathsep.join(
                (str(self.python_packages_dir), existing_pythonpath)
            )
        else:
            merged_env["PYTHONPATH"] = str(self.python_packages_dir)

        self.logger.info("Executing pip command: %s", " ".join(command))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=merged_env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if progress_callback:
                try:
                    progress_callback(line)
                except Exception:  # pragma: no cover - defensive logging
                    self.logger.debug("Progress callback raised an exception", exc_info=True)
            self.logger.info("pip: %s", line)

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"pip exited with status {return_code}")

        self.logger.info("Dependencies installed into %s", self.python_packages_dir)
        self._refresh_environment_notes()
