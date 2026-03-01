from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def ori_vector_from_index(index: str) -> np.ndarray:

    if not isinstance(index, str) or len(index) < 2:
        raise ValueError("index must look like 'a1' or 'b3'.")

    prefix = index[0].lower()
    num = int(index[1:])          # 1-based id in your description
    i = num - 1                   # convert to 0-based for Python

    if prefix == "a":
        if not (0 <= i < len(ORI_2_AY)):
            raise IndexError(f"{index}: id out of range for ORI_2_AY (1..{len(ORI_2_AY)}).")
        M = ORI_2_AY[i]

    elif prefix == "b":
        if not (0 <= i < len(ORI_2_BASE)):
            raise IndexError(f"{index}: id out of range for ORI_2_BASE (1..{len(ORI_2_BASE)}).")
        M = ORI_2_BASE[i]

    else:
        raise ValueError("index prefix must be 'a' or 'b'.")

    return np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]], dtype=float)


# ============================= User Config =============================
ORI_2_AY = [np.array([[0.58, 0.019, -0.015], [0.019, 0.17, -0.012], [-0.015, -0.012, 0.25]]),
            np.array([[0.40, 0.069, 0.26], [0.069, 0.17, -0.001], [0.26, -0.001, 0.43]]),
            np.array([[0.19, 0.028, 0.00], [0.028, 0.81, 0.0], [0.0, 0.0, 0.0]]), ]

ORI_2_BASE = [np.diag([1.0, 0.0, 0.0]), np.diag([1/2, 1/2, 0.0]), np.diag([1/3, 1/3, 1/3]),
              np.diag([2/3, 1/6, 1/6]), np.diag([3/4, 1/4, 0.0]), np.diag([5/12, 5/12, 1/6]), ]

COMPO_ID = 0
INDEX = "a3"
ANALYSIS_TPYE = ["tm", "etc"][0]
ORI_VECTOR = ori_vector_from_index(INDEX)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR
ANALYSIS_NAME = f"Analysis_{INDEX}_{ANALYSIS_TPYE}.daf"

# Material1 (isotropic): keep unchanged
M1_YOUNG = [0.0448, 0.003][COMPO_ID]
M1_POISSON = [0.35, 0.35][COMPO_ID]
M1_THERMAL_EXPANSION = [2.6e-5, 7.0e-5][COMPO_ID]
M1_THERMAL_CONDUCTIVITY = 0.22
M1_SPECIFIC_HEAT_CAPACITY = 1.0

# Material2 (transversely isotropic): keep unchanged
M2_AXIAL_YOUNG = [0.23313, 0.172][COMPO_ID]
M2_INPLANE_YOUNG = [0.02311, 0.172][COMPO_ID]
M2_INPLANE_POISSON = [0.404, 0.20][COMPO_ID]
M2_TRANSVERSE_POISSON = [0.20, 0.20][COMPO_ID]
M2_TRANSVERSE_SHEAR = [0.0897, 0.07167][COMPO_ID]
M2_AXIAL_CTE = [-2.4e-6, 1.0e-6][COMPO_ID]
M2_INPLANE_CTE = [6.4e-6, 1.0e-6][COMPO_ID]
M2_AXIAL_THERMAL_CONDUCTIVITY = 8.8
M2_TRANSVERSE_THERMAL_CONDUCTIVITY = 2.0
M2_SPECIFIC_HEAT_CAPACITY = 1.0

# Adjustable phase / orientation
PHASE2_VOLUME_FRACTION = 0.15
ASPECT_RATIO = 10.0 * 1.25

# Other constants
MATERIAL1_DENSITY = 1.0
MATERIAL2_DENSITY = 1.0
REFERENCE_TEMPERATURE = 0.0
PHASE1_VOLUME_FRACTION = 1.0 - PHASE2_VOLUME_FRACTION


ORI_11, ORI_22, ORI_33, ORI_12, ORI_13, ORI_23 = [float(v) for v in ORI_VECTOR]


TM_MATERIAL1_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Material1"),
    ("type", "elastic"),
    ("density", MATERIAL1_DENSITY),
    ("elastic_model", "isotropic"),
    ("Young", M1_YOUNG),
    ("Poisson", M1_POISSON),
    ("thermal_expansion", M1_THERMAL_EXPANSION),
    ("reference_temperature", REFERENCE_TEMPERATURE),
]

TM_MATERIAL2_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Material2"),
    ("type", "elastic"),
    ("density", MATERIAL2_DENSITY),
    ("elastic_model", "transversely_isotropic"),
    ("axial_Young", M2_AXIAL_YOUNG),
    ("inPlane_Young", M2_INPLANE_YOUNG),
    ("inPlane_Poisson", M2_INPLANE_POISSON),
    ("transverse_Poisson", M2_TRANSVERSE_POISSON),
    ("transverse_shear", M2_TRANSVERSE_SHEAR),
    ("axial_CTE", M2_AXIAL_CTE),
    ("inPlane_CTE", M2_INPLANE_CTE),
    ("reference_temperature", REFERENCE_TEMPERATURE),
]

ETC_MATERIAL1_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Material1"),
    ("type", "linear_fourier"),
    ("density", MATERIAL1_DENSITY),
    ("consistent_tangent", "on"),
    ("thermal_model", "isotropic"),
    ("thermal_conductivity", M1_THERMAL_CONDUCTIVITY),
    ("specific_heat_capacity", M1_SPECIFIC_HEAT_CAPACITY),
]

ETC_MATERIAL2_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Material2"),
    ("type", "linear_fourier"),
    ("density", MATERIAL2_DENSITY),
    ("consistent_tangent", "on"),
    ("thermal_model", "transversely_isotropic"),
    ("specific_heat_capacity", M2_SPECIFIC_HEAT_CAPACITY),
    ("axial_thermal_conductivity", M2_AXIAL_THERMAL_CONDUCTIVITY),
    ("transverse_thermal_conductivity", M2_TRANSVERSE_THERMAL_CONDUCTIVITY),
]

PHASE1_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Phase1"),
    ("type", "matrix"),
    ("volume_fraction", PHASE1_VOLUME_FRACTION),
    ("material", "Material1"),
]

PHASE2_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Phase2"),
    ("type", "inclusion"),
    ("volume_fraction", PHASE2_VOLUME_FRACTION),
    ("behavior", "deformable_solid"),
    ("material", "Material2"),
    ("aspect_ratio", ASPECT_RATIO),
    ("orientation", "tensor"),
    ("orientation_11", ORI_11),
    ("orientation_22", ORI_22),
    ("orientation_33", ORI_33),
    ("orientation_12", ORI_12),
    ("orientation_13", ORI_13),
    ("orientation_23", ORI_23),
    ("closure", "orthotropic"),
    ("coated", "no"),
]

MICROSTRUCTURE_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Microstructure1"),
    ("phase", "Phase1"),
    ("phase", "Phase2"),
]

TM_MECHANICAL_LOADING_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Mechanical"),
    ("type", "strain"),
    ("load", "uniaxial_1"),
    ("initial_strain", 0.0),
    ("peak_strain", 3.0e-2),
    ("history", "monotonic"),
    ("quasi_static", "on"),
    ("theta_load", 90.0),
    ("phi_load", 0.0),
]

TM_TEMPERATURE_LOADING_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Temperature"),
    ("type", "temperature"),
    ("initial_temperature", 0.0),
    ("peak_temperature", 1.0),
    ("history", "monotonic"),
]

ETC_LOADING_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Temperature_gradient"),
    ("type", "temperature_gradient"),
    ("load", "uniaxial_1"),
    ("initial_gradient", 0.0),
    ("peak_gradient", 1.0),
    ("history", "monotonic"),
    ("theta_load", 90.0),
    ("phi_load", 0.0),
]

RVE_PARAMS: list[tuple[str, float | str]] = [
    ("type", "classical"),
    ("microstructure", "Microstructure1"),
]

ANALYSIS_COMMON_PARAMS: list[tuple[str, float | str]] = [
    ("final_time", 1.0),
    ("max_time_inc", 1.0),
    ("min_time_inc", 1.0e-1),
    ("finite_strain", "off"),
    ("output_name", "output1"),
    ("load", "DIGIMAT"),
    ("homogenization", "on"),
    ("homogenization_model", "Mori_Tanaka"),
    ("integration_parameter", 5.0e-1),
    ("number_angle_increments", 6),
    ("output_precision", 5),
    ("homogenization_it_max", 200),
    ("homogenization_monitoring_it", 200),
    ("plane_condition_initial_guess", "on"),
    ("OT_trace_tol", 1.0e-2),
    ("hybrid_methodology", "off"),
    ("hybrid_failure_criteria", "off"),
    ("FPGF_refinement", "on"),
]

OUTPUT_PARAMS: list[tuple[str, float | str]] = [
    ("name", "output1"),
    ("RVE_data", "Default"),
    ("Phase_data", "Phase1,Default"),
    ("Phase_data", "Phase2,Default"),
    ("Engineering_data", "Default"),
    ("Log_data", "Default"),
    ("Dependent_data", "Default"),
    ("Fatigue_data", "Default"),
    ("Composite_data", "None"),
]


# ============================== Internals ==============================
SECTION_MARKER = "##########################################"


def _format_value(value: float | str) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{float(value):.15e}"
    return str(value)


def _render_section(section_name: str, params: Sequence[tuple[str, float | str]]) -> list[str]:
    lines: list[str] = [SECTION_MARKER, section_name]
    for key, value in params:
        lines.append(f"{key} = {_format_value(value)}")
    lines.append("")
    return lines


def _normalize_analysis_type(value: str) -> str:
    analysis_type = str(value).strip().lower()
    if analysis_type not in {"tm", "etc"}:
        raise ValueError(f"Unsupported ANALYSIS_TPYE: {value}. Expected 'tm' or 'etc'.")
    return analysis_type


def _build_analysis_params(analysis_type: str) -> list[tuple[str, float | str]]:
    if analysis_type == "tm":
        header = [
            ("name", "Analysis1"),
            ("type", "thermo_mechanical"),
            ("loading_name", "Mechanical,Temperature"),
        ]
    else:
        header = [
            ("name", "Analysis2"),
            ("type", "thermal_conductivity"),
            ("loading_name", "Temperature_gradient"),
        ]
    return header + ANALYSIS_COMMON_PARAMS


def _build_daf_text(analysis_type: str) -> str:
    analysis_type = _normalize_analysis_type(analysis_type)
    lines: list[str] = []

    if analysis_type == "tm":
        lines += _render_section("MATERIAL", TM_MATERIAL1_PARAMS)
        lines += _render_section("MATERIAL", TM_MATERIAL2_PARAMS)
    else:
        lines += _render_section("MATERIAL", ETC_MATERIAL1_PARAMS)
        lines += _render_section("MATERIAL", ETC_MATERIAL2_PARAMS)

    lines += _render_section("PHASE", PHASE1_PARAMS)
    lines += _render_section("PHASE", PHASE2_PARAMS)
    lines += _render_section("MICROSTRUCTURE", MICROSTRUCTURE_PARAMS)
    if analysis_type == "tm":
        lines += _render_section("LOADING", TM_MECHANICAL_LOADING_PARAMS)
        lines += _render_section("LOADING", TM_TEMPERATURE_LOADING_PARAMS)
    else:
        lines += _render_section("LOADING", ETC_LOADING_PARAMS)
    lines += _render_section("RVE", RVE_PARAMS)
    lines += _render_section("ANALYSIS", _build_analysis_params(analysis_type))
    lines += _render_section("OUTPUT", OUTPUT_PARAMS)
    return "\n".join(lines).rstrip() + "\n"


def generate_one_daf() -> Path:
    analysis_type = _normalize_analysis_type(ANALYSIS_TPYE)
    output_text = _build_daf_text(analysis_type=analysis_type)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = ANALYSIS_NAME if ANALYSIS_NAME else f"Analysis_{INDEX}_{analysis_type}.daf"
    output_path = output_dir / output_name
    output_path.write_text(output_text, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    generated_path = generate_one_daf()
    print("Generated file:")
    print(generated_path)
