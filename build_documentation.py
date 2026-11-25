"""Append current textual project sources to DETAILED_DOCUMENTATION.md."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCUMENT = ROOT / "DETAILED_DOCUMENTATION.md"
MARKER = "## Complete source listings"
SOURCE_FILES = [
    "ai.py",
    "marl.py",
    "prediction.py",
    "main.py",
    "make_datasets.py",
    "train_models.py",
    "train_marl.py",
    "build_documentation.py",
    "tests/test_ai.py",
    "tests/test_marl.py",
    "tests/test_prediction.py",
    "tests/test_training_data.py",
    "pytest.ini",
    "README.md",
    "training_report.json",
]


LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".ini": "ini",
    ".md": "markdown",
    ".json": "json",
}


def main() -> None:
    current = DOCUMENT.read_text(encoding="utf-8")
    if MARKER not in current:
        raise ValueError(f"Documentation marker missing: {MARKER}")
    introduction = current.split(MARKER, 1)[0].rstrip()
    parts = [
        introduction,
        "",
        MARKER,
        "",
        "The following listings are generated verbatim from the current project files. "
        "Binary model checkpoints and CSV datasets are documented by schema instead of embedded.",
        "",
    ]
    for relative_name in SOURCE_FILES:
        path = ROOT / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"Documented source file is missing: {path}")
        language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")
        source = path.read_text(encoding="utf-8").rstrip()
        parts.extend(
            [
                f"### `{relative_name}`",
                "",
                f"````{language}",
                source,
                "````",
                "",
            ]
        )
    DOCUMENT.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"rebuilt: {DOCUMENT}")
    print(f"documented source files: {len(SOURCE_FILES)}")
    print(f"bytes: {DOCUMENT.stat().st_size}")


if __name__ == "__main__":
    main()
