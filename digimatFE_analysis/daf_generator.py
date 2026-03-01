from __future__ import annotations

from pathlib import Path
from typing import Sequence


# ============================= User Config =============================
INDEX = 1
INPUT_DIR = Path("point_angle_files")
OUTPUT_DIR = Path("digimatFE_analysis")

# Material1 (isotropic): adjustable
M1_YOUNG = 1.0
M1_POISSON = 0.25
M1_THERMAL_EXPANSION = 0.0

# Material2 (transversely isotropic): 7 adjustable parameters
M2_AXIAL_YOUNG = 1.0
M2_INPLANE_YOUNG = 0.2
M2_INPLANE_POISSON = 0.2
M2_TRANSVERSE_POISSON = 0.2
M2_TRANSVERSE_SHEAR = 0.25
M2_AXIAL_CTE = 0.0
M2_INPLANE_CTE = 0.0

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


MATERIAL1_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Material1"),
    ("type", "elastic"),
    ("density", MATERIAL1_DENSITY),
    ("elastic_model", "isotropic"),
    ("Young", M1_YOUNG),
    ("Poisson", M1_POISSON),
    ("thermal_expansion", M1_THERMAL_EXPANSION),
    ("reference_temperature", REFERENCE_TEMPERATURE),
]

MATERIAL2_PARAMS: list[tuple[str, float | str]] = [
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

LOADING_PARAMS: list[tuple[str, float | str]] = [
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

TEMPERATURE_LOADING_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Temperature"),
    ("type", "temperature"),
    ("initial_temperature", 0.0),
    ("peak_temperature", 1.0),
    ("history", "monotonic"),
    ("temperature_load_application", "concurrent"),
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

ANALYSISFE_PARAMS: list[tuple[str, float | str]] = [
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
        return f"{float(value):.15e}"
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


def _build_daf_text(positions: list[list[float]], angles: list[list[float]]) -> str:
    lines: list[str] = []
    lines += _render_section("MATERIAL", MATERIAL1_PARAMS)
    lines += _render_section("MATERIAL", MATERIAL2_PARAMS)
    lines += _render_section("PHASE", PHASE1_PARAMS)
    lines += _render_phase2_with_custom(positions, angles)
    lines += _render_section("MICROSTRUCTURE", MICROSTRUCTURE_PARAMS)
    lines += _render_section("LOADING", LOADING_PARAMS)
    lines += _render_section("LOADING", TEMPERATURE_LOADING_PARAMS)
    lines += _render_section("RVE", RVE_PARAMS)
    lines += _render_section("MESH", MESH_PARAMS)
    lines += _render_section("ANALYSISFE", ANALYSISFE_PARAMS)
    lines += _render_section("GLOBAL_SETTINGS", GLOBAL_SETTINGS_PARAMS)
    return "\n".join(lines).rstrip() + "\n"


def generate_one_daf(index: int = INDEX) -> Path:
    angle_path = _resolve_input_file(
        INPUT_DIR,
        (
            f"angles_a{index}.txt",
            f"angles_a{index}",
            f"angle_a{index}.txt",
            f"angle_a{index}",
        ),
    )
    point_path = _resolve_input_file(
        INPUT_DIR,
        (
            f"points_a{index}.txt",
            f"points_a{index}",
            f"point_a{index}.txt",
            f"point_a{index}",
        ),
    )

    positions = _read_table(point_path, min_cols=3)
    angles = _read_table(angle_path, min_cols=2)
    output_text = _build_daf_text(positions, angles)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"Analysis_a{index}.daf"
    output_path.write_text(output_text, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    generated_path = generate_one_daf()
    print("Generated file:")
    print(generated_path)
