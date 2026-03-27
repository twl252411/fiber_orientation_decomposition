from __future__ import annotations

from pathlib import Path
from typing import Sequence


# ============================= User Config =============================
COMPO_ID = 1
INDEXES = ["a1", "a2", "a3", "b1", "b2", "b3", "b4", "b5", "b6"]
ANALYSIS_TPYES = ["tm", "etc"]
# 批量选择：默认使用全部；可改成如 [INDEXES[0]] / [ANALYSIS_TPYES[0]]
SELECTED_INDEXES = INDEXES
SELECTED_ANALYSIS_TPYES = ANALYSIS_TPYES
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "point_angle_files"
OUTPUT_DIR = PROJECT_ROOT / "digimatFE_analysis"

# Material1 (isotropic): adjustable
M1_YOUNG = [0.0448, 0.003][COMPO_ID]
M1_POISSON = [0.35, 0.35][COMPO_ID]
M1_THERMAL_EXPANSION = [2.6E-5, 7.0E-5,][COMPO_ID]
M1_THERMAL_CONDUCTIVITY = 0.22
M1_SPECIFIC_HEAT_CAPACITY = 1.0

# Material2 (transversely isotropic): 7 adjustable parameters
M2_AXIAL_YOUNG = [0.23313, 0.172][COMPO_ID]
M2_INPLANE_YOUNG = [0.02311, 0.172][COMPO_ID]
M2_INPLANE_POISSON = [0.404, 0.20][COMPO_ID]
M2_TRANSVERSE_POISSON = [0.20, 0.20][COMPO_ID]
M2_TRANSVERSE_SHEAR = [0.0897, 0.07167][COMPO_ID]
M2_AXIAL_CTE = [-2.4E-6, 1.0E-6][COMPO_ID]
M2_INPLANE_CTE = [6.4E-6, 1.0E-6][COMPO_ID]
M2_AXIAL_THERMAL_CONDUCTIVITY = 8.8
M2_TRANSVERSE_THERMAL_CONDUCTIVITY = 2.0
M2_SPECIFIC_HEAT_CAPACITY = 1.0

# Phase2 / mesh / rve: adjustable
PHASE2_VOLUME_FRACTION = 0.15
INCLUSION_DIAMETER = 5.0
INCLUSION_SIZE = 50.0
ELEMENT_SIZE = 2.0
MINIMUM_ELEMENT_SIZE = 0.25
RVE_SIZE = 100.0

# Other constants (usually unchanged)
MATERIAL1_DENSITY = 1.0
MATERIAL2_DENSITY = 1.0
REFERENCE_TEMPERATURE = 0.0
PHASE1_VOLUME_FRACTION = 1.0 - PHASE2_VOLUME_FRACTION


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

# ------------------------------- Phases -------------------------------
PHASE1_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Phase1"),
    ("type", "matrix"),
    ("volume_fraction", PHASE1_VOLUME_FRACTION),
    ("material", "Material1"),
]

PHASE2_FIXED_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Phase2"),
    ("type", "inclusion_fe"),
    ("volume_fraction", PHASE2_VOLUME_FRACTION),
    ("material", "Material2"),
    ("inclusion_shape", "cylinder"),
    ("aspect_ratio", 10.0),
    ("phase_definition", "by_size_and_diameter"),
    ("inclusion_diameter", INCLUSION_DIAMETER),
    ("inclusion_size", INCLUSION_SIZE),
    ("size_distribution", "fixed"),
    ("orientation", "fixed"),
    ("theta_angle", 90.0),
    ("phi_angle", 0.0),
    ("coated", "no"),
    ("interface_behavior", "perfectly_bonded"),
    ("clustering", "no"),
    ("allow_size_reduction", "no"),
    ("track_percolation_onset", "no"),
    ("stop_at_percolation", "no"),
    ("check_final_percolation", "no"),
    ("no_tie_on_fiber_tips", "no"),
    ("custom_position_usage", "sequential"),
    ("custom_position_ignore_phase_fraction", "on"),
    ("custom_position_disable_geom_check", "on"),
]

MICROSTRUCTURE_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Microstructure1"),
    ("phase", "Phase1"),
    ("phase", "Phase2"),
]

TM_LOADING_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Mechanical"),
    ("type", "strain"),
    ("boundary_conditions", "periodic"),
    ("custom_bc", "no"),
    ("load", "uniaxial_1"),
    ("initial_strain", 0.0),
    ("peak_strain", 3.0e-2),
    ("history", "monotonic"),
    ("quasi_static", "on"),
    ("theta_load", 90.0),
    ("phi_load", 0.0),
    ("required_components", "E1_E2_E3_G12_G23_G13_CTE123_"),
]

TM_TEMPERATURE_LOADING_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Temperature"),
    ("type", "temperature"),
    ("initial_temperature", 0.0),
    ("peak_temperature", 1.0),
    ("history", "monotonic"),
    ("temperature_load_application", "concurrent"),
]

ETC_LOADING_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Temperature_gradient"),
    ("type", "temperature_gradient"),
    ("boundary_conditions", "periodic"),
    ("load", "uniaxial_1"),
    ("initial_gradient", 0.0),
    ("peak_gradient", 1.0),
    ("history", "monotonic"),
    ("theta_load", 90.0),
    ("phi_load", 0.0),
    ("required_components", "K11_K22_K33_"),
]

RVE_PARAMS: list[tuple[str, float | str]] = [
    ("type", "classical"),
    ("microstructure", "Microstructure1"),
]

MESH_PARAMS: list[tuple[str, float | str]] = [
    ("mesh_type", "conforming"),
    ("automatic_mesh_sizing", "off"),
    ("element_size", ELEMENT_SIZE),
    ("minimum_element_size", MINIMUM_ELEMENT_SIZE),
    ("use_quadratic_elements", "off"),
    ("use_quadratic_geometric_elements", "off"),
    ("element_shape", "quad_dominated"),
    ("internal_coarsening", "on"),
    ("curvature_control", "on"),
    ("chordal_deviation_ratio", 1.0e-1),
    ("nb_refinement_steps", 5),
    ("model_layer_interfaces", "off"),
    ("seed_size", 5.0),
    ("share_nodes", "off"),
    ("periodic_mesh", "off"),
    ("cohesive_element_size_ratio", 2.0e-1),
]

TM_ANALYSISFE_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Analysis1"),
    ("type", "thermo_mechanical"),
    ("loading_name", "Mechanical,Temperature"),
    ("final_time", 1.0),
    ("max_time_inc", 1.0),
    ("min_time_inc", 1.0e-1),
    ("finite_strain", "off"),
    ("initial_time_inc", 1.0),
    ("max_number_increment", 2),
    ("rve_size_definition", "user_defined"),
    ("rve_dimension", "3d"),
    ("size_rve", RVE_SIZE),
    ("periodic", "yes"),
    ("generation_sequence", "proportional"),
    ("generate_matrix", "no"),
    ("track_global_percolation_onset", "no"),
    ("stop_at_global_percolation", "no"),
    ("check_final_global_percolation", "no"),
    ("random_seed_type", "automatic"),
    ("random_seed", -75161927),
    ("fe_solver", "Abaqus/INP"),
    ("unsymmetric_solver", "no"),
    ("default_timestepping", "yes"),
    ("nb_cpus", 8),
    ("fe_solver_type", "iterative"),
    ("fe_field_output_frequency", 1),
    ("use_output_time_points", "yes"),
]

ETC_ANALYSISFE_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Analysis1"),
    ("type", "thermal_conductivity"),
    ("loading_name", "Temperature_gradient"),
    ("final_time", 1.0),
    ("max_time_inc", 1.0),
    ("min_time_inc", 1.0e-1),
    ("finite_strain", "off"),
    ("initial_time_inc", 1.0),
    ("max_number_increment", 2),
    ("rve_size_definition", "user_defined"),
    ("rve_dimension", "3d"),
    ("size_rve", RVE_SIZE),
    ("periodic", "yes"),
    ("generation_sequence", "proportional"),
    ("generate_matrix", "no"),
    ("track_global_percolation_onset", "no"),
    ("stop_at_global_percolation", "no"),
    ("check_final_global_percolation", "no"),
    ("random_seed_type", "automatic"),
    ("random_seed", -75161927),
    ("fe_solver", "Abaqus/INP"),
    ("unsymmetric_solver", "no"),
    ("default_timestepping", "yes"),
    ("nb_cpus", 8),
    ("fe_solver_type", "iterative"),
    ("fe_field_output_frequency", 1),
    ("use_output_time_points", "yes"),
]

GLOBAL_SETTINGS_PARAMS: list[tuple[str, float | str]] = [
    ("allow_interpenetration", "no"),
    ("allow_coating_interpenetration", "no"),
    ("allow_rim_interpenetration", "no"),
    ("use_median_plane_interpenetration", "no"),
    ("cubic_architecture", "no"),
    ("apply_perturbation", "no"),
    ("favor_orientation_over_fraction", "no"),
    ("minimum_relative_distance_wrt_diameter", 5.0e-2),
    ("minimum_relative_vol", 5.0e-2),
    ("max_number_of_tests", 2000),
    ("OT_norm_tol", 1.0e-1),
    ("max_number_of_geometry_attempts", 10),
    ("minimum_rel_dist_incl_to_face", 0.0),
    ("maximum_interpenetration_amount", 1.0),
    ("random_fiber_perturbation_no_transverse_perturbation", "no"),
    ("default_geometric_options", "yes"),
    ("remove_unconnected_matrix_regions", "no"),
]


# ============================== Internals ==============================
SECTION_MARKER = "##########################################"


def _format_value(value: float | str) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{float(value):.12e}"
    return str(value)


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
        input_dir = Path(input_dir)
        candidate = input_dir / name
        if candidate.exists():
            return candidate
    tried = ", ".join(str(input_dir / n) for n in name_candidates)
    raise FileNotFoundError(f"Cannot find any input file. Tried: {tried}")


def _render_section(section_name: str, params: Sequence[tuple[str, float | str]]) -> list[str]:
    lines: list[str] = [SECTION_MARKER, section_name]
    for key, value in params:
        lines.append(f"{key} = {_format_value(value)}")
    lines.append("")
    return lines


def _render_phase2_with_custom(positions: list[list[float]], angles: list[list[float]]) -> list[str]:
    if len(positions) != len(angles):
        raise ValueError(
            f"Point/orientation count mismatch: {len(positions)} points vs {len(angles)} angles."
        )

    lines = _render_section("PHASE", PHASE2_FIXED_PARAMS)
    lines.pop()  # remove trailing blank, append custom lines first

    for row in positions:
        if len(row) < 3:
            raise ValueError("Point rows must contain at least 3 columns.")
        lines.append(
            "custom_position = "
            f"{_format_value(row[0])} ; {_format_value(row[1])} ; {_format_value(row[2])}"
        )

    for row in angles:
        if len(row) < 2:
            raise ValueError("Angle rows must contain at least 2 columns.")
        lines.append(
            "custom_orientation = "
            f"{_format_value(row[1])} ; {_format_value(row[0])} ; {_format_value(0.0)}"
        )

    lines.append("")
    return lines


def _normalize_analysis_type(value: str) -> str:
    analysis_type = str(value).strip().lower()
    if analysis_type not in {"tm", "etc"}:
        raise ValueError(f"Unsupported ANALYSIS_TPYE: {value}. Expected 'tm' or 'etc'.")
    return analysis_type


def _build_daf_text(positions: list[list[float]], angles: list[list[float]], analysis_type: str) -> str:
    analysis_type = _normalize_analysis_type(analysis_type)
    lines: list[str] = []
    if analysis_type == "tm":
        lines += _render_section("MATERIAL", TM_MATERIAL1_PARAMS)
        lines += _render_section("MATERIAL", TM_MATERIAL2_PARAMS)
    else:
        lines += _render_section("MATERIAL", ETC_MATERIAL1_PARAMS)
        lines += _render_section("MATERIAL", ETC_MATERIAL2_PARAMS)

    lines += _render_section("PHASE", PHASE1_PARAMS)
    lines += _render_phase2_with_custom(positions, angles)
    lines += _render_section("MICROSTRUCTURE", MICROSTRUCTURE_PARAMS)
    if analysis_type == "tm":
        lines += _render_section("LOADING", TM_LOADING_PARAMS)
        lines += _render_section("LOADING", TM_TEMPERATURE_LOADING_PARAMS)
    else:
        lines += _render_section("LOADING", ETC_LOADING_PARAMS)
    lines += _render_section("RVE", RVE_PARAMS)
    lines += _render_section("MESH", MESH_PARAMS)
    if analysis_type == "tm":
        lines += _render_section("ANALYSISFE", TM_ANALYSISFE_PARAMS)
    else:
        lines += _render_section("ANALYSISFE", ETC_ANALYSISFE_PARAMS)
    lines += _render_section("GLOBAL_SETTINGS", GLOBAL_SETTINGS_PARAMS)
    return "\n".join(lines).rstrip() + "\n"


def _normalize_indexes(indexes: Sequence[str]) -> list[str]:
    normalized = [str(idx).strip() for idx in indexes if str(idx).strip()]
    if not normalized:
        raise ValueError("SELECTED_INDEXES is empty.")
    return normalized


def _normalize_analysis_types(analysis_types: Sequence[str]) -> list[str]:
    normalized = [_normalize_analysis_type(v) for v in analysis_types]
    if not normalized:
        raise ValueError("SELECTED_ANALYSIS_TPYES is empty.")
    return normalized


def generate_one_daf(
    index: str,
    analysis_type: str,
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    analysis_type = _normalize_analysis_type(analysis_type)
    index = str(index).strip()
    if not index:
        raise ValueError("index is empty.")

    point_file = f"points_{index}.txt"
    angle_file = f"angles_{index}.txt"

    angle_path = _resolve_input_file(input_dir, (angle_file,))
    point_path = _resolve_input_file(input_dir, (point_file,))

    positions = _read_table(point_path, min_cols=3)
    angles = _read_table(angle_path, min_cols=2)
    output_text = _build_daf_text(positions, angles, analysis_type=analysis_type)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"Analysis_{index}_{analysis_type}.daf"
    output_path = output_dir / output_name
    output_path.write_text(output_text, encoding="utf-8")
    return output_path


def generate_batch_dafs(
    indexes: Sequence[str] = SELECTED_INDEXES,
    analysis_types: Sequence[str] = SELECTED_ANALYSIS_TPYES,
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    normalized_indexes = _normalize_indexes(indexes)
    normalized_analysis_types = _normalize_analysis_types(analysis_types)

    generated: list[Path] = []
    for index in normalized_indexes:
        for analysis_type in normalized_analysis_types:
            generated.append(
                generate_one_daf(
                    index=index,
                    analysis_type=analysis_type,
                    input_dir=input_dir,
                    output_dir=output_dir,
                )
            )
    return generated


if __name__ == "__main__":
    generated_paths = generate_batch_dafs()
    print("Generated DAF files:")
    for path in generated_paths:
        print(path)
