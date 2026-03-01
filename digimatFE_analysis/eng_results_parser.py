from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _format_float(value: float) -> str:
    return f"{float(value):.15e}"


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
    eng_path = input_dir / f"Analysis_a{index}.eng"
    if not eng_path.exists():
        raise FileNotFoundError(f"ENG file not found: {eng_path}")

    stiffness, cte = parse_eng_file(eng_path)

    out_dir = input_dir if output_dir is None else output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stiffness_path = out_dir / f"Analysis_a{index}_Stiffness.txt"
    cte_path = out_dir / f"Analysis_a{index}_CTE.txt"

    _write_csv_rows(stiffness_path, stiffness)
    _write_csv_rows(cte_path, cte)
    return stiffness_path, cte_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Stiffness Matrix and Thermal Expansion from Digimat-FE .eng file."
    )
    parser.add_argument("--index", type=int, required=True, help="Index x for Analysis_a{x}.eng.")
    parser.add_argument("--input-dir", type=Path, default=Path("digimatFE_analysis"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to input-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stiffness_path, cte_path = extract_and_save(
        index=args.index,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print("Saved:")
    print(stiffness_path)
    print(cte_path)


if __name__ == "__main__":
    main()
