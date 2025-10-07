"""Runtime dependency audit utilities.

Este módulo consolida verificações de ambiente em torno dos arquivos de
dependências oficiais do projeto (`requirements*.txt`). O objetivo é
detectar bibliotecas ausentes, versões fora das políticas declaradas e
possíveis divergências de hashes declarativos, produzindo um relatório
estruturado reutilizável pela UI e pelos logs do aplicativo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence
import json
import logging
import os
import shlex
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LOGGER = logging.getLogger(__name__)


REQUIREMENT_FILES: tuple[Path, ...] = (
    Path("requirements.txt"),
    Path("requirements-optional.txt"),
    Path("requirements-extras.txt"),
    Path("requirements-test.txt"),
)


@dataclass(slots=True)
class DependencyIssue:
    """Representa um problema detectado durante a auditoria."""

    name: str
    requirement_string: str
    requirement_file: str
    line_number: int
    installed_version: str | None
    specifier: str
    marker: str | None
    extras: tuple[str, ...]
    category: str
    suggestion: str
    hashes: tuple[str, ...] = ()
    details: Mapping[str, object] | None = None


@dataclass(slots=True)
class DependencyAuditResult:
    """Estrutura consolidada com o resultado da auditoria."""

    generated_at: datetime
    environment: Mapping[str, str]
    files_scanned: tuple[str, ...]
    missing: list[DependencyIssue] = field(default_factory=list)
    version_mismatches: list[DependencyIssue] = field(default_factory=list)
    hash_mismatches: list[DependencyIssue] = field(default_factory=list)

    def has_issues(self) -> bool:
        return bool(self.missing or self.version_mismatches or self.hash_mismatches)

    def summary_message(self) -> str:
        missing = len(self.missing)
        mismatched = len(self.version_mismatches)
        hash_mismatched = len(self.hash_mismatches)
        if missing == mismatched == hash_mismatched == 0:
            return "Dependency audit completed — no issues detected."
        parts: list[str] = ["Dependency audit completed"]
        issues: list[str] = []
        if missing:
            issues.append(f"{missing} missing")
        if mismatched:
            issues.append(f"{mismatched} version conflict(s)")
        if hash_mismatched:
            issues.append(f"{hash_mismatched} hash divergence(s)")
        parts.append("; ".join(issues))
        return " — ".join(parts)

    def to_serializable(self) -> dict[str, object]:
        """Transforma o relatório em um dicionário serializável."""

        def _serialize_issue(issue: DependencyIssue) -> dict[str, object]:
            payload: dict[str, object] = {
                "name": issue.name,
                "requirement": issue.requirement_string,
                "file": issue.requirement_file,
                "line": issue.line_number,
                "installed_version": issue.installed_version,
                "specifier": issue.specifier,
                "marker": issue.marker,
                "extras": list(issue.extras),
                "category": issue.category,
                "suggestion": issue.suggestion,
                "hashes": list(issue.hashes),
            }
            if issue.details:
                payload["details"] = dict(issue.details)
            return payload

        return {
            "generated_at": self.generated_at.isoformat(),
            "environment": dict(self.environment),
            "files_scanned": list(self.files_scanned),
            "missing": [_serialize_issue(item) for item in self.missing],
            "version_mismatches": [
                _serialize_issue(item) for item in self.version_mismatches
            ],
            "hash_mismatches": [
                _serialize_issue(item) for item in self.hash_mismatches
            ],
        }


@dataclass(slots=True)
class _ParsedRequirement:
    requirement: Requirement
    original_text: str
    requirement_file: Path
    line_number: int
    hashes: tuple[str, ...]


def _iter_requirement_lines(path: Path) -> Iterator[tuple[int, str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for idx, raw_line in enumerate(handle, start=1):
                yield idx, raw_line.rstrip("\n")
    except FileNotFoundError:
        LOGGER.debug("Requirement file missing: %s", path)


def _normalize_tokens(tokens: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    requirement_tokens: list[str] = []
    hashes: list[str] = []
    skip_next_hash_value = False
    for token in tokens:
        if skip_next_hash_value:
            hashes.append(token)
            skip_next_hash_value = False
            continue
        lowered = token.lower()
        if lowered == "--hash":
            skip_next_hash_value = True
            continue
        if lowered.startswith("--hash="):
            _, _, value = token.partition("=")
            if value:
                hashes.append(value)
            continue
        if lowered.startswith("--"):
            # Ignore other pip options in requirement files.
            continue
        requirement_tokens.append(token)
    requirement_str = " ".join(requirement_tokens).strip()
    return requirement_str, tuple(hashes)


def _parse_requirement_line(
    line: str,
    *,
    requirement_file: Path,
    line_number: int,
) -> _ParsedRequirement | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith(("-r", "--requirement", "-c", "--constraint")):
        LOGGER.debug(
            "Ignoring recursive include/constraint directive in %s:%d: %s",
            requirement_file,
            line_number,
            stripped,
        )
        return None

    lexer = shlex.shlex(stripped, posix=True)
    lexer.commenters = "#"
    lexer.whitespace_split = True
    tokens = list(lexer)
    requirement_str, hashes = _normalize_tokens(tokens)
    if not requirement_str:
        return None

    try:
        requirement = Requirement(requirement_str)
    except Exception as exc:
        LOGGER.warning(
            "Failed to parse requirement '%s' in %s:%d: %s",
            requirement_str,
            requirement_file,
            line_number,
            exc,
        )
        return None

    return _ParsedRequirement(
        requirement=requirement,
        original_text=requirement_str,
        requirement_file=requirement_file,
        line_number=line_number,
        hashes=hashes,
    )


def _iter_requirements(paths: Iterable[Path]) -> Iterator[_ParsedRequirement]:
    for path in paths:
        for line_number, raw in _iter_requirement_lines(path):
            parsed = _parse_requirement_line(
                raw,
                requirement_file=path,
                line_number=line_number,
            )
            if parsed is not None:
                yield parsed


def _evaluate_marker(requirement: Requirement) -> bool:
    if requirement.marker is None:
        return True
    env = default_environment()
    try:
        return bool(requirement.marker.evaluate(env))
    except Exception as exc:
        LOGGER.debug(
            "Failed to evaluate marker '%s' for requirement '%s': %s",
            requirement.marker,
            requirement,
            exc,
        )
        return True


def _get_distribution(
    requirement: Requirement,
) -> tuple[importlib_metadata.Distribution | None, str | None]:
    package_name = canonicalize_name(requirement.name)
    try:
        dist = importlib_metadata.distribution(package_name)
    except PackageNotFoundError:
        return None, None
    except Exception:
        LOGGER.exception("Unexpected error while probing distribution for %s", package_name)
        return None, None
    return dist, dist.version


def _normalized_requirement_string(requirement: Requirement) -> str:
    extras = ""
    if requirement.extras:
        extras = f"[{','.join(sorted(requirement.extras))}]"
    specifier = str(requirement.specifier) if requirement.specifier else ""
    if requirement.url:
        base = f"{requirement.name} @ {requirement.url}"
        return f"{base}{specifier}" if specifier else base
    return f"{requirement.name}{extras}{specifier}"


def _build_install_command(parsed: _ParsedRequirement) -> str:
    requirement = parsed.requirement
    requirement_no_marker = Requirement(str(requirement))
    requirement_no_marker.marker = None
    return f"python -m pip install \"{_normalized_requirement_string(requirement_no_marker)}\""


def _extract_archive_hashes(
    distribution: importlib_metadata.Distribution,
) -> tuple[str, ...]:
    """Obtém hashes de `direct_url.json` quando disponíveis."""

    try:
        raw_payload = distribution.read_text("direct_url.json")
    except FileNotFoundError:
        return ()
    except Exception:
        LOGGER.debug(
            "Unable to read direct_url.json for %s", distribution.metadata["Name"], exc_info=True
        )
        return ()
    if not raw_payload:
        return ()
    try:
        data = json.loads(raw_payload)
    except json.JSONDecodeError:
        LOGGER.debug(
            "direct_url.json for %s is not valid JSON",
            distribution.metadata.get("Name", "unknown"),
            exc_info=True,
        )
        return ()
    archive_info = data.get("archive_info")
    if not isinstance(archive_info, dict):
        return ()
    hash_value = archive_info.get("hash")
    if not isinstance(hash_value, str):
        return ()
    if "=" in hash_value:
        _, _, hash_value = hash_value.partition("=")
    return (hash_value.lower(),)


def _classify_requirement(parsed: _ParsedRequirement) -> DependencyIssue | None:
    requirement = parsed.requirement
    if not _evaluate_marker(requirement):
        LOGGER.debug("Marker for %s evaluated to False; skipping.", requirement)
        return None

    distribution, installed_version = _get_distribution(requirement)
    requirement_string = _normalized_requirement_string(requirement)
    suggestion = _build_install_command(parsed)
    marker_text = str(requirement.marker) if requirement.marker is not None else None
    extras_tuple = tuple(sorted(requirement.extras)) if requirement.extras else ()
    specifier_text = str(requirement.specifier) if requirement.specifier else ""

    if distribution is None:
        return DependencyIssue(
            name=requirement.name,
            requirement_string=requirement_string,
            requirement_file=str(parsed.requirement_file),
            line_number=parsed.line_number,
            installed_version=None,
            specifier=specifier_text,
            marker=marker_text,
            extras=extras_tuple,
            category="missing",
            suggestion=suggestion,
            hashes=parsed.hashes,
        )

    if specifier_text and requirement.specifier and not requirement.specifier.contains(
        installed_version or "", prereleases=True
    ):
        return DependencyIssue(
            name=requirement.name,
            requirement_string=requirement_string,
            requirement_file=str(parsed.requirement_file),
            line_number=parsed.line_number,
            installed_version=installed_version,
            specifier=specifier_text,
            marker=marker_text,
            extras=extras_tuple,
            category="version_mismatch",
            suggestion=suggestion,
            hashes=parsed.hashes,
            details={"installed": installed_version},
        )

    if parsed.hashes:
        expected_hashes = {value.lower() for value in parsed.hashes}
        actual_hashes = set(_extract_archive_hashes(distribution))
        if actual_hashes and actual_hashes.isdisjoint(expected_hashes):
            return DependencyIssue(
                name=requirement.name,
                requirement_string=requirement_string,
                requirement_file=str(parsed.requirement_file),
                line_number=parsed.line_number,
                installed_version=installed_version,
                specifier=specifier_text,
                marker=marker_text,
                extras=extras_tuple,
                category="hash_mismatch",
                suggestion=suggestion,
                hashes=parsed.hashes,
                details={
                    "expected": sorted(expected_hashes),
                    "detected": sorted(actual_hashes),
                },
            )

    return None


def audit_environment(paths: Iterable[Path] = REQUIREMENT_FILES) -> DependencyAuditResult:
    """Executa a auditoria dos arquivos de dependência configurados."""

    parsed_requirements = list(_iter_requirements(paths))
    missing: list[DependencyIssue] = []
    mismatched: list[DependencyIssue] = []
    hash_mismatched: list[DependencyIssue] = []

    for parsed in parsed_requirements:
        issue = _classify_requirement(parsed)
        if issue is None:
            continue
        if issue.category == "missing":
            missing.append(issue)
        elif issue.category == "version_mismatch":
            mismatched.append(issue)
        elif issue.category == "hash_mismatch":
            hash_mismatched.append(issue)

    env_snapshot = {
        "python_version": sys.version.split()[0],
        "platform": os.name,
        "cwd": str(Path.cwd()),
    }

    return DependencyAuditResult(
        generated_at=datetime.now(timezone.utc),
        environment=env_snapshot,
        files_scanned=tuple(str(path) for path in paths),
        missing=missing,
        version_mismatches=mismatched,
        hash_mismatches=hash_mismatched,
    )


__all__ = [
    "DependencyAuditResult",
    "DependencyIssue",
    "audit_environment",
]

