#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import pathlib
from typing import Iterable, List, Sequence, Tuple

LOG_METHOD_NAMES = {"debug", "info", "warning", "error", "critical", "exception", "log"}


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
    files = discover_targets(args.roots)
    if not files:
        print("No Python files found under the requested roots.")
        return 0
    base = pathlib.Path.cwd()
    any_findings = False
    for path in sorted(files):
        findings = scan_file(path, include_all=args.all_strings)
        rel_path = to_ascii(str(path.relative_to(base)))
        header_printed = False
        if findings["logging"]:
            if not header_printed:
                print(f"\nFile: {rel_path}")
                header_printed = True
            print("  Non ASCII logging strings:")
            for lineno, text, preview in findings["logging"]:
                print(f"    line {lineno}: {text}")
                if preview:
                    print(f"      preview: {preview}")
            any_findings = True
        if args.all_strings and findings["literals"]:
            if not header_printed:
                print(f"\nFile: {rel_path}")
                header_printed = True
            print("  Other non ASCII literals:")
            for lineno, text, preview in findings["literals"]:
                print(f"    line {lineno}: {text}")
                if preview:
                    print(f"      preview: {preview}")
            any_findings = True
        notes: List[str] = []
        if findings["decode_issue"]:
            notes.append(f"decode issue: {findings['decode_issue']}")
        if findings["null_bytes"]:
            notes.append("contains NUL bytes")
        if findings["utf16"]:
            if not header_printed:
                print(f"\nFile: {rel_path}")
                header_printed = True
            print("  UTF-16 markers:")
            for lineno, preview in findings["utf16"]:
                print(f"    line {lineno}: {preview}")
        if notes:
            if not header_printed:
                print(f"\nFile: {rel_path}")
                header_printed = True
            print("  Notes:")
            for note in notes:
                print(f"    - {note}")
    if any_findings:
        print("\nScan finished with non ASCII strings detected.")
    else:
        print("Scan finished without non ASCII logging strings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
