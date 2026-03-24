from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence


# ============================= User Config =============================
INDEX = "a2"
ANALYSIS_TYPE = ["tm", "etc"][0]
INPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR: Path | None = None
TMP_DIR = f"tmp_{INDEX}_{ANALYSIS_TYPE}"
TMP_DIR_GLOB = "tmp_*_*"


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


def _shape(rows: Sequence[Sequence[float]]) -> tuple[int, int]:
    if not rows:
        return (0, 0)
    return (len(rows), len(rows[0]))


def _flatten(rows: Sequence[Sequence[float]]) -> list[float]:
    return [v for row in rows for v in row]


def _ensure_3x3_for_etc(raw: list[list[float]]) -> list[list[float]]:
    if _shape(raw) != (3, 3):
        raise ValueError(f"ETC matrix must be 3x3, got shape {_shape(raw)}.")
    return raw


def _cte_1x6_to_3x3(raw: list[list[float]]) -> list[list[float]]:
    # Input order: [11,22,33,12,23,13]
    # Output 3x3 order (rows/cols 1,2,3):
    # [[11,12,13],
    #  [12,22,23],
    #  [13,23,33]]
    if _shape(raw) == (3, 3):
        return raw

    flat = _flatten(raw)
    if len(flat) != 6:
        raise ValueError(f"CTE must be 1x6 (or 6 values), got shape {_shape(raw)}.")
    c11, c22, c33, c12, c23, c13 = flat
    return [
        [c11, c12, c13],
        [c12, c22, c23],
        [c13, c23, c33],
    ]


def _stiffness_reorder_6x6(raw: list[list[float]]) -> list[list[float]]:
    # Input row/col order:  [11,22,33,12,23,13]
    # Output row/col order: [11,22,33,12,13,23]
    if _shape(raw) != (6, 6):
        raise ValueError(f"Stiffness must be 6x6, got shape {_shape(raw)}.")
    perm = [0, 1, 2, 3, 5, 4]
    return [[raw[i][j] for j in perm] for i in perm]


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


def _parse_tmp_dir_name(name: str) -> tuple[str, str] | None:
    # tmp_{index}_{analysis_type}, analysis_type in {tm,etc}
    match = re.fullmatch(r"tmp_(.+)_(tm|etc)", name.strip().lower())
    if not match:
        return None
    return match.group(1), match.group(2)


def _resolve_eng_path(tmp_dir_path: Path, index_tag: str, analysis_type: str) -> Path:
    candidates = [
        tmp_dir_path / f"Analysis_{index_tag}_{analysis_type}.eng",
        tmp_dir_path / f"Analysis_{index_tag}_{analysis_type}_Analysis1.eng",
        tmp_dir_path / f"Analysis_{index_tag}_{analysis_type}_Analysis2.eng",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    fallback = sorted(tmp_dir_path.glob("*.eng"), key=lambda p: p.stat().st_mtime, reverse=True)
    if fallback:
        return fallback[0]
    return candidates[0]


def parse_tm(eng_path: Path) -> tuple[list[list[float]], list[list[float]]]:
    lines = eng_path.read_text(encoding="utf-8").splitlines()
    stiffness_raw = _extract_numeric_block(lines, "Stiffness Matrix in Global Axes")
    cte_raw = _extract_numeric_block(lines, "Thermal Expansion in Global Axes")
    stiffness = _stiffness_reorder_6x6(stiffness_raw)
    cte = _cte_1x6_to_3x3(cte_raw)
    return stiffness, cte


def parse_etc(eng_path: Path) -> list[list[float]]:
    lines = eng_path.read_text(encoding="utf-8").splitlines()
    etc_raw = _extract_numeric_block_with_candidates(
        lines,
        titles=[
            "Thermal Conductivity Tensor in Global Axes",
            "Thermal Conductivity Matrix in Global Axes",
            "Conductivity Matrix in Global Axes",
        ],
    )
    return _ensure_3x3_for_etc(etc_raw)


def extract_and_save(
    index: str = INDEX,
    analysis_type: str = ANALYSIS_TYPE,
    input_dir: Path = INPUT_DIR,
    output_dir: Path | None = OUTPUT_DIR,
    tmp_dir: str | None = TMP_DIR,
) -> list[Path]:
    analysis_type = _normalize_analysis_type(analysis_type)
    index_tag = str(index).strip()
    if not index_tag:
        raise ValueError("INDEX is empty.")

    tmp_dir_name = tmp_dir if tmp_dir else f"tmp_{index_tag}_{analysis_type}"
    tmp_dir_path = Path(input_dir) / tmp_dir_name
    eng_path = _resolve_eng_path(tmp_dir_path=tmp_dir_path, index_tag=index_tag, analysis_type=analysis_type)
    if not eng_path.exists():
        raise FileNotFoundError(f"ENG file not found: {eng_path}")
    print(f"Using ENG file: {eng_path}")

    out_dir = Path(input_dir) if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"Analysis_{index_tag}_{analysis_type}"

    if analysis_type == "tm":
        stiffness, cte = parse_tm(eng_path)
        stiffness_path = out_dir / f"{prefix}_Stiffness.txt"
        cte_path = out_dir / f"{prefix}_CTE.txt"
        _write_csv_rows(stiffness_path, stiffness)
        _write_csv_rows(cte_path, cte)
        return [stiffness_path, cte_path]

    conductivity = parse_etc(eng_path)
    conductivity_path = out_dir / f"{prefix}_Conductivity.txt"
    _write_csv_rows(conductivity_path, conductivity)
    return [conductivity_path]


def extract_all_and_save(
    input_dir: Path = INPUT_DIR,
    output_dir: Path | None = OUTPUT_DIR,
    tmp_dir_glob: str = TMP_DIR_GLOB,
) -> list[Path]:
    saved: list[Path] = []
    root = Path(input_dir)
    for tmp_dir in sorted(root.glob(tmp_dir_glob)):
        if not tmp_dir.is_dir():
            continue
        parsed = _parse_tmp_dir_name(tmp_dir.name)
        if parsed is None:
            continue
        index_tag, analysis_type = parsed
        saved.extend(
            extract_and_save(
                index=index_tag,
                analysis_type=analysis_type,
                input_dir=root,
                output_dir=output_dir,
                tmp_dir=tmp_dir.name,
            )
        )
    return saved


def main() -> None:
    saved_paths = extract_all_and_save(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        tmp_dir_glob=TMP_DIR_GLOB,
    )
    print("Saved:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
