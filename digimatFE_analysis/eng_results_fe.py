from __future__ import annotations

from pathlib import Path
from typing import Sequence
import os


# ============================= User Config =============================
INDEX = "b3"
ANALYSIS_TYPE = ["tm", "etc"][1]
TMP_DIR = f"tmp_{INDEX}_{ANALYSIS_TYPE}"
JAB_NAME = f"Analysis_{INDEX}_{ANALYSIS_TYPE}"
PROJECT_ROOT = r"H:\\github\\fiber_orientation_decomposition"
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

        row = _parse_numeric_line(line)
        block.append(row)
        started = True

    if not block:
        raise ValueError(f"No numeric data found in section '{title}'.")
    return block


def _write_csv_rows(path: Path, rows: Sequence[Sequence[float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(_format_float(v) for v in row))
            f.write("\n")


def parse_eng_file(eng_path: Path) -> tuple[list[list[float]], list[list[float]]]:
    lines = eng_path.read_text(encoding="utf-8").splitlines()
    stiffness = _extract_numeric_block(lines, "Stiffness Matrix in Global Axes")
    cte = _extract_numeric_block(lines, "Thermal Expansion in Global Axes")
    return stiffness, cte


def extract_and_save(
    index: int,
    input_dir: Path = Path("digimatFE_analysis"),
    output_dir: Path | None = None,
) -> tuple[Path, Path]:

    eng_path = Path(input_dir) / TMP_DIR / f"{JAB_NAME}_Analysis1.eng"
    if not eng_path.exists():
        raise FileNotFoundError(f"ENG file not found: {eng_path}")

    stiffness, cte = parse_eng_file(eng_path)

    out_dir = input_dir if output_dir is None else output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stiffness_path = Path(out_dir) / f"Analysis_a{index}_Stiffness.txt"
    cte_path = Path(out_dir) / f"Analysis_a{index}_CTE.txt"

    _write_csv_rows(stiffness_path, stiffness)
    _write_csv_rows(cte_path, cte)
    return stiffness_path, cte_path


def main() -> None:
    stiffness_path, cte_path = extract_and_save(
        index=INDEX,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )
    print("Saved:")
    print(stiffness_path)
    print(cte_path)


if __name__ == "__main__":
    main()
