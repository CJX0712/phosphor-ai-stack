"""P0 source guard: no symbol literals, no BOM, no tabs in python sources.

Emoji and pictograph literals are banned because a source file saved through a
GBK codepage mangles them into multi byte garbage that then breaks regex
character classes at import time. The check is written with ord() ranges only,
so the guard itself can never be corrupted the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path

BANNED_RANGES = (
    (0x2190, 0x21FF),  # arrows
    (0x2300, 0x23FF),  # misc technical
    (0x2460, 0x24FF),  # enclosed alphanumerics
    (0x2500, 0x27BF),  # box drawing .. dingbats
    (0x2B00, 0x2BFF),  # misc symbols and arrows
    (0x1F000, 0x1FAFF),  # emoji
    (0xFE00, 0xFE0F),  # variation selectors
    (0x200D, 0x200D),  # zero width joiner
)

ALLOWED_EXTRA = {0x2026}  # ellipsis used by truncation helpers

SCAN_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".html", ".json"}


def scan_file(path: Path) -> list[str]:
    findings: list[str] = []
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        findings.append(f"{path}: BOM detected")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        findings.append(f"{path}: not valid utf-8 ({exc})")
        return findings
    for lineno, line in enumerate(text.splitlines(), start=1):
        for ch in line:
            cp = ord(ch)
            if cp in ALLOWED_EXTRA:
                continue
            for low, high in BANNED_RANGES:
                if low <= cp <= high:
                    findings.append(f"{path}:{lineno}: U+{cp:04X} symbol literal")
                    break
    if path.suffix == ".py" and "\t" in text:
        findings.append(f"{path}: tab character in python source")
    return findings


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        parts = set(path.parts)
        if {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".artifacts",
            "dist",
            "build",
        } & parts:
            continue
        findings.extend(scan_file(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent
    findings = scan(root)
    if findings:
        print(f"P0 GUARD FAILED: {len(findings)} finding(s)")
        for item in findings[:50]:
            print("  " + item)
        return 1
    print("P0 GUARD OK: no symbol literals, no BOM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
