from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


# ============================== CONFIG ==============================
BASELINE_MAT_FILE = SCRIPT_DIR / "Template.mat"
OUTPUT_DIR = SCRIPT_DIR

COMPO_ID = 0
DEFAULT_INDEX = "b1"
DEFAULT_ANALYSIS_TYPE = ["tm", "etc"][0]

# Adjustable defaults
PHASE2_VOLUME_FRACTION = 0.15
ASPECT_RATIO = 12.5

# Keep these material presets unchanged.
MATERIAL_PRESETS: list[dict[str, float]] = [
    {
        "m1_young": 0.0448,
        "m1_poisson": 0.35,
        "m1_thermal_expansion": 2.6e-5,
        "m2_axial_young": 0.23313,
        "m2_inplane_young": 0.02311,
        "m2_inplane_poisson": 0.404,
        "m2_transverse_poisson": 0.20,
        "m2_transverse_shear": 0.0897,
        "m2_axial_cte": -2.4e-6,
        "m2_inplane_cte": 6.4e-6,
    },
    {
        "m1_young": 0.003,
        "m1_poisson": 0.35,
        "m1_thermal_expansion": 7.0e-5,
        "m2_axial_young": 0.172,
        "m2_inplane_young": 0.172,
        "m2_inplane_poisson": 0.20,
        "m2_transverse_poisson": 0.20,
        "m2_transverse_shear": 0.07167,
        "m2_axial_cte": 1.0e-6,
        "m2_inplane_cte": 1.0e-6,
    },
]

# Orientation tensors
ORI_2_AY = [
    np.array([[0.58, 0.019, -0.015], [0.019, 0.17, -0.012], [-0.015, -0.012, 0.25]]),
    np.array([[0.40, 0.069, 0.26], [0.069, 0.17, -0.001], [0.26, -0.001, 0.43]]),
    np.array([[0.19, 0.028, 0.00], [0.028, 0.81, 0.0], [0.0, 0.0, 0.0]]),
]

ORI_2_BASE = [
    np.diag([1.0, 0.0, 0.0]),
    np.diag([1.0 / 2.0, 1.0 / 2.0, 0.0]),
    np.diag([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    np.diag([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]),
    np.diag([3.0 / 4.0, 1.0 / 4.0, 0.0]),
    np.diag([5.0 / 12.0, 5.0 / 12.0, 1.0 / 6.0]),
]

# Batch generation list:
# - orientation_vector can override index-derived orientation.
# - output_name can override default Analysis_{index}_{analysis_type}.mat.
# - template_file can override BASELINE_MAT_FILE for this case only.
BATCH_CASES: list[dict[str, Any]] = [
    {
        "index": DEFAULT_INDEX,
        "analysis_type": DEFAULT_ANALYSIS_TYPE,
        "compo_id": COMPO_ID,
        "phase2_volume_fraction": PHASE2_VOLUME_FRACTION,
        "aspect_ratio": ASPECT_RATIO,
    }
]


ReplacementKey = tuple[str, int, str]
SECTION_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def ori_vector_from_index(index: str) -> np.ndarray:
    if not isinstance(index, str) or len(index) < 2:
        raise ValueError("index must look like 'a1' or 'b3'.")

    prefix = index[0].lower()
    item_index = int(index[1:]) - 1

    if prefix == "a":
        if not (0 <= item_index < len(ORI_2_AY)):
            raise IndexError(f"{index}: id out of range for ORI_2_AY (1..{len(ORI_2_AY)}).")
        tensor = ORI_2_AY[item_index]
    elif prefix == "b":
        if not (0 <= item_index < len(ORI_2_BASE)):
            raise IndexError(f"{index}: id out of range for ORI_2_BASE (1..{len(ORI_2_BASE)}).")
        tensor = ORI_2_BASE[item_index]
    else:
        raise ValueError("index prefix must be 'a' or 'b'.")

    return np.array(
        [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]],
        dtype=float,
    )


def _normalize_analysis_type(value: str) -> str:
    analysis_type = str(value).strip().lower()
    if analysis_type not in {"tm", "etc"}:
        raise ValueError(f"Unsupported analysis_type: {value}. Expected 'tm' or 'etc'.")
    return analysis_type


def _format_value(value: float | int | str) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{float(value):.15e}"
    return str(value)


def _is_section_header(line: str) -> bool:
    return bool(SECTION_PATTERN.fullmatch(line.strip()))


def inspect_template_fields(template_path: Path = BASELINE_MAT_FILE) -> dict[str, list[str]]:
    text = Path(template_path).read_text(encoding="utf-8")
    section_counts: dict[str, int] = {}
    current_section = ""
    fields: dict[str, list[str]] = {}

    for raw in text.splitlines():
        stripped = raw.strip()
        if _is_section_header(stripped):
            current_section = stripped
            section_counts[current_section] = section_counts.get(current_section, 0) + 1
            fields[f"{current_section}#{section_counts[current_section]}"] = []
            continue
        if "=" not in raw or not current_section:
            continue
        key = raw.split("=", 1)[0].strip()
        block_key = f"{current_section}#{section_counts[current_section]}"
        fields[block_key].append(key)
    return fields


def _build_case_replacements(case: Mapping[str, Any]) -> dict[ReplacementKey, float | str]:
    index = str(case.get("index", DEFAULT_INDEX))
    analysis_type = _normalize_analysis_type(str(case.get("analysis_type", DEFAULT_ANALYSIS_TYPE)))
    compo_id = int(case.get("compo_id", COMPO_ID))
    if compo_id < 0 or compo_id >= len(MATERIAL_PRESETS):
        raise ValueError(f"compo_id out of range: {compo_id}")

    preset = MATERIAL_PRESETS[compo_id]
    phase2_vf = float(case.get("phase2_volume_fraction", PHASE2_VOLUME_FRACTION))
    phase1_vf = 1.0 - phase2_vf
    aspect_ratio = float(case.get("aspect_ratio", ASPECT_RATIO))

    orientation_vector = case.get("orientation_vector")
    if orientation_vector is None:
        orientation_vector = ori_vector_from_index(index)
    if len(orientation_vector) != 6:
        raise ValueError("orientation_vector must contain 6 values: [11,22,33,12,13,23].")
    ori_11, ori_22, ori_33, ori_12, ori_13, ori_23 = [float(v) for v in orientation_vector]

    replacements: dict[ReplacementKey, float | str] = {
        ("MATERIAL", 1, "Young"): float(case.get("m1_young", preset["m1_young"])),
        ("MATERIAL", 1, "Poisson"): float(case.get("m1_poisson", preset["m1_poisson"])),
        (
            "MATERIAL",
            1,
            "thermal_expansion",
        ): float(case.get("m1_thermal_expansion", preset["m1_thermal_expansion"])),
        ("MATERIAL", 2, "axial_Young"): float(case.get("m2_axial_young", preset["m2_axial_young"])),
        ("MATERIAL", 2, "inPlane_Young"): float(case.get("m2_inplane_young", preset["m2_inplane_young"])),
        (
            "MATERIAL",
            2,
            "inPlane_Poisson",
        ): float(case.get("m2_inplane_poisson", preset["m2_inplane_poisson"])),
        (
            "MATERIAL",
            2,
            "transverse_Poisson",
        ): float(case.get("m2_transverse_poisson", preset["m2_transverse_poisson"])),
        (
            "MATERIAL",
            2,
            "transverse_shear",
        ): float(case.get("m2_transverse_shear", preset["m2_transverse_shear"])),
        ("MATERIAL", 2, "axial_CTE"): float(case.get("m2_axial_cte", preset["m2_axial_cte"])),
        ("MATERIAL", 2, "inPlane_CTE"): float(case.get("m2_inplane_cte", preset["m2_inplane_cte"])),
        ("PHASE", 1, "volume_fraction"): phase1_vf,
        ("PHASE", 2, "volume_fraction"): phase2_vf,
        ("PHASE", 2, "aspect_ratio"): aspect_ratio,
        ("PHASE", 2, "orientation_11"): ori_11,
        ("PHASE", 2, "orientation_22"): ori_22,
        ("PHASE", 2, "orientation_33"): ori_33,
        ("PHASE", 2, "orientation_12"): ori_12,
        ("PHASE", 2, "orientation_13"): ori_13,
        ("PHASE", 2, "orientation_23"): ori_23,
    }

    if analysis_type == "tm":
        replacements[("ANALYSIS", 1, "name")] = "Analysis1"
        replacements[("ANALYSIS", 1, "type")] = "thermo_mechanical"
        replacements[("ANALYSIS", 1, "loading_name")] = "Mechanical,Temperature"
    else:
        replacements[("ANALYSIS", 1, "name")] = "Analysis2"
        replacements[("ANALYSIS", 1, "type")] = "thermal_conductivity"
        replacements[("ANALYSIS", 1, "loading_name")] = "Temperature_gradient"

    return replacements


def _apply_replacements(template_text: str, replacements: Mapping[ReplacementKey, float | str]) -> str:
    section_counts: dict[str, int] = {}
    current_section = ""
    output_lines: list[str] = []

    for raw in template_text.splitlines():
        stripped = raw.strip()

        if _is_section_header(stripped):
            current_section = stripped
            section_counts[current_section] = section_counts.get(current_section, 0) + 1
            output_lines.append(raw)
            continue

        if "=" in raw and current_section:
            left, _right = raw.split("=", 1)
            key = left.strip()
            block_idx = section_counts[current_section]
            replacement_key = (current_section, block_idx, key)
            if replacement_key in replacements:
                output_lines.append(f"{left.rstrip()} = {_format_value(replacements[replacement_key])}")
                continue

        output_lines.append(raw)

    return "\n".join(output_lines).rstrip() + "\n"


def _output_name(case: Mapping[str, Any]) -> str:
    if "output_name" in case and str(case["output_name"]).strip():
        return str(case["output_name"])
    index = str(case.get("index", DEFAULT_INDEX))
    analysis_type = _normalize_analysis_type(str(case.get("analysis_type", DEFAULT_ANALYSIS_TYPE)))
    return f"Analysis_{index}_{analysis_type}.mat"


def generate_one_mat(
    case: Mapping[str, Any],
    baseline_mat_file: Path = BASELINE_MAT_FILE,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    baseline_mat_file = Path(baseline_mat_file)
    if not baseline_mat_file.exists():
        raise FileNotFoundError(f"Baseline MAT not found: {baseline_mat_file}")

    template_text = baseline_mat_file.read_text(encoding="utf-8")
    replacements = _build_case_replacements(case)
    rendered = _apply_replacements(template_text, replacements)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _output_name(case)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def generate_batch_mats(
    cases: list[Mapping[str, Any]] = BATCH_CASES,
    baseline_mat_file: Path = BASELINE_MAT_FILE,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    paths: list[Path] = []
    for case in cases:
        case_template = Path(case["template_file"]) if "template_file" in case else Path(baseline_mat_file)
        paths.append(generate_one_mat(case=case, baseline_mat_file=case_template, output_dir=output_dir))
    return paths


def main() -> None:
    generated_paths = generate_batch_mats()
    print("Generated MAT files:")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
