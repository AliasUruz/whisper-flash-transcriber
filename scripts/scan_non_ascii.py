#!/usr/bin/env python3
from __future__ import annotations

"""Scanner para identificar strings não ASCII, com saída via logging estruturado."""

import argparse
import ast
import logging
import pathlib
from typing import Iterable, List, Sequence, Tuple

from src.logging_utils import get_logger, log_context, log_operation, setup_logging

LOG_METHOD_NAMES = {"debug", "info", "warning", "error", "critical", "exception", "log"}

LOGGER = get_logger(
    "whisper_flash_transcriber.scripts.scan_non_ascii",
    component="ScanNonASCII",
    default_event="scripts.scan_non_ascii",
)


def parse_args() -> argparse.Namespace:
    """Configura os argumentos da CLI, orientando o uso de logs estruturados."""

    parser = argparse.ArgumentParser(
        description="Scan Python files for logging strings that contain non ASCII characters.",
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=["src", "tests", "scripts"],
        help="Files or directories to scan. Defaults to src, tests, and scripts.",
    )
    parser.add_argument(
        "--all-strings",
        action="store_true",
        help="Include non ASCII string literals outside logging calls.",
    )
    return parser.parse_args()


def discover_targets(roots: Sequence[str]) -> List[pathlib.Path]:
    files: List[pathlib.Path] = []
    seen = set()
    base = pathlib.Path.cwd()
    for entry in roots:
        path = pathlib.Path(entry)
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            continue
        if path.is_dir():
            for candidate in path.rglob("*.py"):
                if candidate not in seen:
                    files.append(candidate)
                    seen.add(candidate)
        elif path.suffix == ".py":
            if path not in seen:
                files.append(path)
                seen.add(path)
    return files


def has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def joined_static_text(node: ast.JoinedStr) -> str:
    parts: List[str] = []
    for value in node.values:
        if isinstance(value, ast.Str):
            parts.append(value.s)
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
    return "".join(parts)


def iter_string_nodes(items: Iterable[ast.AST]) -> Iterable[ast.AST]:
    for item in items:
        if isinstance(item, (ast.Str, ast.JoinedStr)):
            yield item
        elif isinstance(item, ast.Constant) and isinstance(item.value, str):
            yield item
        elif isinstance(item, ast.List):
            yield from iter_string_nodes(item.elts)
        elif isinstance(item, ast.Tuple):
            yield from iter_string_nodes(item.elts)
        elif isinstance(item, ast.Dict):
            yield from iter_string_nodes(list(item.keys) + list(item.values))


def is_logging_call(func: ast.AST) -> bool:
    if isinstance(func, ast.Attribute):
        return func.attr in LOG_METHOD_NAMES
    return False


def to_ascii(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


def scan_file(path: pathlib.Path, include_all: bool) -> dict:
    logging_hits: List[Tuple[int, str, str]] = []
    literal_hits: List[Tuple[int, str, str]] = []
    decode_issue = None
    utf16_lines: List[Tuple[int, str]] = []
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return {
            "logging": logging_hits,
            "literals": literal_hits,
            "decode_issue": f"read error: {exc}",
            "null_bytes": False,
            "utf16": utf16_lines,
        }
    null_bytes = b"\x00" in raw_bytes
    try:
        source = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decode_issue = "utf-8"
        source = raw_bytes.decode("utf-8", errors="replace")
    lower_source = source.lower()
    if "utf-16" in lower_source or "utf16" in lower_source:
        for idx, line in enumerate(source.splitlines(), start=1):
            low = line.lower()
            if "utf-16" in low or "utf16" in low:
                utf16_lines.append((idx, to_ascii(line.strip())))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "logging": logging_hits,
            "literals": literal_hits,
            "decode_issue": f"syntax error: {exc}",
            "null_bytes": null_bytes,
            "utf16": utf16_lines,
        }
    lines = source.splitlines()
    logging_string_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and is_logging_call(node.func):
            items = list(node.args) + [kw.value for kw in node.keywords]
            for string_node in iter_string_nodes(items):
                if isinstance(string_node, ast.JoinedStr):
                    text = joined_static_text(string_node)
                elif isinstance(string_node, ast.Constant):
                    text = string_node.value
                elif isinstance(string_node, ast.Str):
                    text = string_node.s
                else:
                    text = ""
                if text and has_non_ascii(text):
                    lineno = getattr(string_node, "lineno", getattr(node, "lineno", 0))
                    preview = ""
                    if 0 < lineno <= len(lines):
                        preview = to_ascii(lines[lineno - 1].strip())
                    logging_hits.append((lineno, to_ascii(text), preview))
                logging_string_nodes.add(id(string_node))
    if include_all:
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                text = joined_static_text(node)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
            elif isinstance(node, ast.Str):
                text = node.s
            else:
                continue
            if not text or id(node) in logging_string_nodes:
                continue
            if has_non_ascii(text):
                lineno = getattr(node, "lineno", 0)
                preview = ""
                if 0 < lineno <= len(lines):
                    preview = to_ascii(lines[lineno - 1].strip())
                literal_hits.append((lineno, to_ascii(text), preview))
    return {
        "logging": logging_hits,
        "literals": literal_hits,
        "decode_issue": decode_issue,
        "null_bytes": null_bytes,
        "utf16": utf16_lines,
    }


def main() -> int:
    """Execute a varredura reportando achados via o logger estruturado."""

    setup_logging()
    args = parse_args()

    with log_operation(
        LOGGER,
        "Iniciando varredura por strings não ASCII.",
        event="scripts.scan_non_ascii.run",
        details={
            "root_count": len(args.roots),
            "include_all_literals": bool(args.all_strings),
        },
    ) as operation_id:
        files = discover_targets(args.roots)
        if not files:
            LOGGER.info(
                log_context(
                    "Nenhum arquivo Python encontrado para análise.",
                    event="scripts.scan_non_ascii.no_targets",
                    operation_id=operation_id,
                    roots=[to_ascii(str(root)) for root in args.roots],
                )
            )
            return 0

        base = pathlib.Path.cwd()
        any_findings = False
        totals = {
            "files": len(files),
            "logging_hits": 0,
            "literal_hits": 0,
            "utf16_markers": 0,
        }

        for path in sorted(files):
            findings = scan_file(path, include_all=args.all_strings)
            rel_path = to_ascii(str(path.relative_to(base)))
            file_details = {
                "path": rel_path,
                "operation_id": operation_id,
            }

            if findings["logging"]:
                any_findings = True
                totals["logging_hits"] += len(findings["logging"])
                for lineno, text, preview in findings["logging"]:
                    LOGGER.warning(
                        log_context(
                            "String não ASCII utilizada em chamada de logging.",
                            event="scripts.scan_non_ascii.logging_hit",
                            **file_details,
                            line=lineno,
                            literal=text,
                            preview=preview or None,
                        )
                    )

            if args.all_strings and findings["literals"]:
                any_findings = True
                totals["literal_hits"] += len(findings["literals"])
                for lineno, text, preview in findings["literals"]:
                    LOGGER.warning(
                        log_context(
                            "Literal não ASCII detectado fora de chamadas de logging.",
                            event="scripts.scan_non_ascii.literal_hit",
                            **file_details,
                            line=lineno,
                            literal=text,
                            preview=preview or None,
                        )
                    )

            if findings["utf16"]:
                totals["utf16_markers"] += len(findings["utf16"])
                for lineno, preview in findings["utf16"]:
                    LOGGER.info(
                        log_context(
                            "Marcador UTF-16 identificado no arquivo.",
                            event="scripts.scan_non_ascii.utf16_marker",
                            **file_details,
                            line=lineno,
                            preview=preview,
                        )
                    )

            if findings["decode_issue"] or findings["null_bytes"]:
                LOGGER.warning(
                    log_context(
                        "Anomalia de leitura detectada durante a varredura.",
                        event="scripts.scan_non_ascii.file_note",
                        **file_details,
                        decode_issue=findings["decode_issue"],
                        contains_null_bytes=findings["null_bytes"],
                    )
                )

        summary_event = "scripts.scan_non_ascii.summary"
        LOGGER.log(
            logging.WARNING if any_findings else logging.INFO,
            log_context(
                "Varredura concluída.",
                event=summary_event,
                operation_id=operation_id,
                findings_detected=any_findings,
                totals=totals,
            ),
        )

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
