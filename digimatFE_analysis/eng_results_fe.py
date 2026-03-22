from __future__ import annotations

from pathlib import Path
from typing import Sequence


# ============================= User Config =============================
INDEX = "b6"
ANALYSIS_TYPE = ["tm", "etc"][0]
INPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR: Path | None = None


def _format_float(value: float) -> str:
    return f"{float(value):.4e}"


def _parse_numeric_line(line: str) -> list[float]:
    return [float(part) for part in line.strip().split()]


def _find_section_start(lines: Sequence[str], title: str) -> int:
    target = f"# {title} :"
    for idx, line in enumerate(lines):
        if line.strip() == target:
            return idx
    raise ValueError(f"Section '{title}' not found.")


def _extract_numeric_block(lines: Sequence[str], title: str) -> list[list[float]]:
    section_idx = _find_section_start(lines, title)
    block: list[list[float]] = []
    started = False

    for raw in lines[section_idx + 1:]:
        line = raw.strip()
        if not line:
            if started:
                break
            continue
        if line.startswith("#"):
            if started:
                break
            continue
        block.append(_parse_numeric_line(line))
        started = True

    if not block:
        raise ValueError(f"No numeric data found in section '{title}'.")
    return block


def _extract_numeric_block_with_candidates(lines: Sequence[str], titles: Sequence[str]) -> list[list[float]]:
    last_error: Exception | None = None
    for title in titles:
        try:
            return _extract_numeric_block(lines, title)
        except ValueError as exc:
            last_error = exc
    if last_error:
        raise ValueError(str(last_error))
    raise ValueError("No candidate section titles provided.")


def _write_csv_rows(path: Path, rows: Sequence[Sequence[float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(_format_float(v) for v in row))
            f.write("\n")


def _normalize_analysis_type(value: str) -> str:
    analysis_type = str(value).strip().lower()
    if analysis_type not in {"tm", "etc"}:
        raise ValueError(f"Unsupported ANALYSIS_TYPE: {value}. Expected 'tm' or 'etc'.")
    return analysis_type


def _eng_path(index_tag: str, analysis_type: str, input_dir: Path) -> Path:
    return Path(input_dir) / f"Analysis_{index_tag}_{analysis_type}.eng"


def parse_tm(eng_path: Path) -> tuple[list[list[float]], list[list[float]]]:
    lines = eng_path.read_text(encoding="utf-8").splitlines()
    stiffness = _extract_numeric_block(lines, "Stiffness Matrix in Global Axes")
    cte = _extract_numeric_block(lines, "Thermal Expansion in Global Axes")
    return stiffness, cte


def parse_etc(eng_path: Path) -> list[list[float]]:
    lines = eng_path.read_text(encoding="utf-8").splitlines()
    return _extract_numeric_block_with_candidates(
        lines,
        titles=[
            "Thermal Conductivity Matrix in Global Axes",
            "Conductivity Matrix in Global Axes",
        ],
    )


def extract_and_save(
    index: str = INDEX,
    analysis_type: str = ANALYSIS_TYPE,
    input_dir: Path = INPUT_DIR,
    output_dir: Path | None = OUTPUT_DIR,
) -> list[Path]:
    analysis_type = _normalize_analysis_type(analysis_type)
    index_tag = str(index).strip()
    if not index_tag:
        raise ValueError("INDEX is empty.")

    eng_path = _eng_path(index_tag=index_tag, analysis_type=analysis_type, input_dir=Path(input_dir))
    if not eng_path.exists():
        raise FileNotFoundError(f"ENG file not found: {eng_path}")
    print(f"Using ENG file: {eng_path}")

    out_dir = Path(input_dir) if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if analysis_type == "tm":
        stiffness, cte = parse_tm(eng_path)
        stiffness_path = out_dir / f"Analysis_{index_tag}_Stiffness.txt"
        cte_path = out_dir / f"Analysis_{index_tag}_CTE.txt"
        _write_csv_rows(stiffness_path, stiffness)
        _write_csv_rows(cte_path, cte)
        return [stiffness_path, cte_path]

    conductivity = parse_etc(eng_path)
    conductivity_path = out_dir / f"Analysis_{index_tag}_Conductivity.txt"
    _write_csv_rows(conductivity_path, conductivity)
    return [conductivity_path]


def main() -> None:
    saved_paths = extract_and_save(
        index=INDEX,
        analysis_type=ANALYSIS_TYPE,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )
    print("Saved:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
