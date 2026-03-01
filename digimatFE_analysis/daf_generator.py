from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SECTION_MARKER = "##########################################"


@dataclass(frozen=True)
class DAFParameters:
    material_density: float = 1.0
    material2_density: float | None = None
    young_modulus: float = 1.0
    poisson_ratio: float = 0.25
    phase2_diameter: float = 5.0
    phase2_size: float = 50.0
    element_size: float = 2.0
    minimum_element_size: float = 0.25

    @property
    def resolved_material2_density(self) -> float:
        return self.material_density if self.material2_density is None else self.material2_density


def _format_float(value: float) -> str:
    return f"{float(value):.15e}"


def _read_table(path: Path, min_cols: int) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            parts = [p.strip() for p in text.split(",")]
            if len(parts) < min_cols:
                raise ValueError(f"{path} line {line_no} has {len(parts)} columns, expected at least {min_cols}.")
            try:
                rows.append([float(p) for p in parts])
            except ValueError as exc:
                raise ValueError(f"{path} line {line_no} contains non-numeric data: {text}") from exc

    if not rows:
        raise ValueError(f"{path} is empty.")
    return rows


def _resolve_input_file(input_dir: Path, name_candidates: Sequence[str]) -> Path:
    for name in name_candidates:
        candidate = input_dir / name
        if candidate.exists():
            return candidate
    tried = ", ".join(str(input_dir / n) for n in name_candidates)
    raise FileNotFoundError(f"Cannot find any input file. Tried: {tried}")


def _find_line(lines: Sequence[str], exact_text: str) -> int:
    for idx, line in enumerate(lines):
        if line.strip() == exact_text:
            return idx
    raise ValueError(f"Line not found: {exact_text}")


def _find_next_section(lines: Sequence[str], start_idx: int) -> int:
    for idx in range(start_idx, len(lines)):
        if lines[idx].strip() == SECTION_MARKER:
            return idx
    return len(lines)


def _replace_key_in_block(lines: list[str], key: str, value: float, block_start: int, block_end: int) -> None:
    key_prefix = f"{key} ="
    for idx in range(block_start, block_end):
        if lines[idx].strip().startswith(key_prefix):
            lines[idx] = f"{key} = {_format_float(value)}"
            return
    raise ValueError(f"Cannot find key '{key}' in block {block_start}:{block_end}")


def _replace_custom_lines(lines: list[str], positions: list[list[float]], angles: list[list[float]]) -> None:
    anchor = _find_line(lines, "custom_position_disable_geom_check = on")
    start = anchor + 1
    end = _find_next_section(lines, start)

    if len(positions) != len(angles):
        raise ValueError(
            f"Point/orientation count mismatch: {len(positions)} points vs {len(angles)} angles."
        )

    new_lines: list[str] = []
    for row in positions:
        if len(row) < 3:
            raise ValueError("Point rows must contain at least 3 columns.")
        new_lines.append(
            "custom_position = "
            f"{_format_float(row[0])} ; {_format_float(row[1])} ; {_format_float(row[2])}"
        )

    for row in angles:
        if len(row) < 2:
            raise ValueError("Angle rows must contain at least 2 columns.")
        # custom_orientation first two columns use angle col2, col1 (swap)
        new_lines.append(
            "custom_orientation = "
            f"{_format_float(row[1])} ; {_format_float(row[0])} ; {_format_float(0.0)}"
        )

    lines[start:end] = new_lines + [""]


def _update_template_content(template_text: str, params: DAFParameters, positions: list[list[float]], angles: list[list[float]]) -> str:
    lines = template_text.splitlines()

    material1_name_idx = _find_line(lines, "name = Material1")
    material1_end = _find_next_section(lines, material1_name_idx + 1)
    _replace_key_in_block(lines, "density", params.material_density, material1_name_idx, material1_end)
    _replace_key_in_block(lines, "Young", params.young_modulus, material1_name_idx, material1_end)
    _replace_key_in_block(lines, "Poisson", params.poisson_ratio, material1_name_idx, material1_end)

    material2_name_idx = _find_line(lines, "name = Material2")
    material2_end = _find_next_section(lines, material2_name_idx + 1)
    _replace_key_in_block(lines, "density", params.resolved_material2_density, material2_name_idx, material2_end)

    phase2_name_idx = _find_line(lines, "name = Phase2")
    phase2_end = _find_next_section(lines, phase2_name_idx + 1)
    _replace_key_in_block(lines, "inclusion_diameter", params.phase2_diameter, phase2_name_idx, phase2_end)
    _replace_key_in_block(lines, "inclusion_size", params.phase2_size, phase2_name_idx, phase2_end)

    mesh_idx = _find_line(lines, "MESH")
    mesh_end = _find_next_section(lines, mesh_idx + 1)
    _replace_key_in_block(lines, "element_size", params.element_size, mesh_idx, mesh_end)
    _replace_key_in_block(lines, "minimum_element_size", params.minimum_element_size, mesh_idx, mesh_end)

    _replace_custom_lines(lines, positions, angles)
    return "\n".join(lines) + "\n"


def generate_analysis_file(
    index: int,
    input_dir: Path,
    output_dir: Path,
    template_path: Path,
    params: DAFParameters,
) -> Path:
    angle_path = _resolve_input_file(
        input_dir,
        (
            f"angles_a{index}.txt",
            f"angles_a{index}",
            f"angle_a{index}.txt",
            f"angle_a{index}",
        ),
    )
    point_path = _resolve_input_file(
        input_dir,
        (
            f"points_a{index}.txt",
            f"points_a{index}",
            f"point_a{index}.txt",
            f"point_a{index}",
        ),
    )

    positions = _read_table(point_path, min_cols=3)
    angles = _read_table(angle_path, min_cols=2)

    template_text = template_path.read_text(encoding="utf-8")
    output_text = _update_template_content(template_text, params, positions, angles)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"Analysis_a{index}.daf"
    output_path.write_text(output_text, encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Digimat-FE thermo-mechanical Analysis_a{x}.daf from one point/angle file pair."
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index x for one input pair: angles_a{x} + point(s)_a{x}.",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("point_angle_files"))
    parser.add_argument("--output-dir", type=Path, default=Path("digimatFE_analysis"))
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("digimatFE_analysis/Analysis_a0.daf"),
        help="Thermo-mechanical DAF template (default: Analysis_a0.daf).",
    )

    parser.add_argument("--material-density", type=float, default=1.0)
    parser.add_argument(
        "--material2-density",
        type=float,
        default=None,
        help="Material2 density; defaults to material-density if omitted.",
    )
    parser.add_argument("--young-modulus", type=float, default=1.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.25)
    parser.add_argument("--phase2-diameter", type=float, default=5.0)
    parser.add_argument("--phase2-size", type=float, default=50.0)
    parser.add_argument("--element-size", type=float, default=2.0)
    parser.add_argument("--minimum-element-size", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = DAFParameters(
        material_density=args.material_density,
        material2_density=args.material2_density,
        young_modulus=args.young_modulus,
        poisson_ratio=args.poisson_ratio,
        phase2_diameter=args.phase2_diameter,
        phase2_size=args.phase2_size,
        element_size=args.element_size,
        minimum_element_size=args.minimum_element_size,
    )
    output_path = generate_analysis_file(
        index=args.index,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        template_path=args.template,
        params=params,
    )

    print("Generated file:")
    print(output_path)


if __name__ == "__main__":
    main()
