# ==================================================================================================== #
# main.py — Interpolate property tensors from states b1~b6 (txt inputs only)
# ==================================================================================================== #
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import eigen_utils as eu
import quadratic_interpolation as qi
import tensor_utils as tu


STATE_ORDER = ["b1", "b2", "b3", "b4", "b5", "b6"]
LINEAR_NODE_IDS = ["b1", "b2", "b3"]

ORIENTATION_CASES = [
    np.array([[0.58, 0.019, -0.015], [0.019, 0.17, -0.012], [-0.015, -0.012, 0.25]], dtype=float),
    np.array([[0.40, 0.069, 0.26], [0.069, 0.17, -0.001], [0.26, -0.001, 0.43]], dtype=float),
    np.array([[0.19, 0.028, 0.00], [0.028, 0.81, 0.0], [0.0, 0.0, 0.0]], dtype=float),
]

# -------------------------- User selections -------------------------- #
# Choose source directory:
SOURCE = ["fe", "mf"][0]

# Choose interpolation type:
INTERP = ["linear", "quadratic"][0]

# Choose output property types:
ANALYSIS_TYPES = [
    ["elastic", "cte"],
    ["etc"],
    ["elastic", "cte", "etc"],
][0]

# Choose orientation case: 1, 2, 3
ORIENTATION_CASE = [1, 2, 3][1]

# Output index in Analysis_{Index}_{ANALYSIS_TYPE}.
# If None, it is set automatically from INTERP ("linear" or "quadratic").
OUTPUT_INDEX: str | None = None

# Optional overrides
DATA_DIR_OVERRIDE: Path | None = None
OUT_DIR = Path(__file__).resolve().parents[1] / "orien_decomp_results"

# Optional explicit state labels
# - For linear: ["b1", "b2", "b3"]
# - For quadratic: ["b1", "b2", "b3", "b4", "b5", "b6"]
# - Use None to apply defaults from INTERP
INDICES: List[str] | None = None


def weighted_sum(weights: Sequence[float], tensors: Sequence[np.ndarray]) -> np.ndarray:
    """Weighted sum of same-shaped tensors."""
    if len(weights) != len(tensors):
        raise ValueError(f"weights length ({len(weights)}) != tensors length ({len(tensors)})")
    return sum(w * t for w, t in zip(weights, tensors))


def _to_state_label(index: str) -> str:
    state = index.strip().lower()
    if state not in STATE_ORDER:
        raise ValueError(f"Unsupported state '{index}', expected one of {STATE_ORDER}.")
    return state


def _state_num(index: str) -> str:
    return _to_state_label(index)[1:]


def _resolve_data_dir(source: str, data_dir_override: Path | None) -> Path:
    if data_dir_override is not None:
        data_dir = data_dir_override.resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")
        return data_dir

    repo_root = Path(__file__).resolve().parents[1]
    if source == "fe":
        data_dir = repo_root / "digimatFE_analysis"
    elif source == "mf":
        data_dir = repo_root / "digimatMF_analysis"
    else:
        raise ValueError("SOURCE must be 'fe' or 'mf'.")

    if not data_dir.exists():
        raise FileNotFoundError(f"Selected data directory does not exist: {data_dir}")
    return data_dir


def _first_existing_file(candidates: Iterable[Path]) -> Path:
    tried: List[Path] = []
    for path in candidates:
        tried.append(path)
        if path.exists():
            return path
    candidate_text = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(f"No matching file found. Tried:\n{candidate_text}")


def _load_txt_matrix(path: Path) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"NaN detected when reading: {path}")
    return np.asarray(arr, dtype=float)


def _stiffness_from_input(raw: np.ndarray) -> np.ndarray:
    """Parse stiffness input into 4th-order tensor.

    Supported shapes:
    - 6x6 full Voigt matrix
    - length-9 independent components [C11,C12,C13,C22,C23,C33,C44,C55,C66]
    """
    arr = np.asarray(raw, dtype=float)

    if arr.shape == (6, 6):
        return tu.tensor_voigt(arr)

    flat = arr.reshape(-1)
    if flat.size == 9:
        c11, c12, c13, c22, c23, c33, c44, c55, c66 = flat
        voigt = np.array(
            [
                [c11, c12, c13, 0.0, 0.0, 0.0],
                [c12, c22, c23, 0.0, 0.0, 0.0],
                [c13, c23, c33, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, c44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, c55, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, c66],
            ],
            dtype=float,
        )
        return tu.tensor_voigt(voigt)

    raise ValueError(f"Unsupported stiffness shape {arr.shape}; expected (6,6) or length-9 vector.")


def _symmetric_3x3_from_voigt6(raw: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)

    if arr.shape == (3, 3):
        return arr

    flat = arr.reshape(-1)
    if flat.size != 6:
        raise ValueError(f"Unsupported {name} shape {arr.shape}; expected (3,3) or length-6 vector.")

    xx, yy, zz, xy, xz, yz = flat
    return np.array(
        [
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ],
        dtype=float,
    )


def _cte_1x6_to_3x3(raw: np.ndarray) -> np.ndarray:
    """Convert CTE [11,22,33,12,13,23] (1x6) to symmetric 3x3."""
    flat = np.asarray(raw, dtype=float).reshape(-1)
    if flat.size != 6:
        raise ValueError(f"CTE must be 1x6 (or 6 values), got shape {np.asarray(raw).shape}.")
    c11, c22, c33, c12, c13, c23 = flat
    return np.array(
        [
            [c11, c12, c13],
            [c12, c22, c23],
            [c13, c23, c33],
        ],
        dtype=float,
    )


def _tm_file_candidates(index: str, quantity: str) -> List[str]:
    """Return candidate filenames for tm quantities.

    quantity in {'Stiffness', 'CTE'}.
    """
    idx = _to_state_label(index)
    num = _state_num(idx)
    legacy_ab = f"ab{num}"  # existing FE exports: Analysis_ab1_Stiffness.txt

    return [
        f"Analysis_{idx}_tm_{quantity}.txt",
        f"Analysis_{idx}_{quantity}.txt",
        f"Analysis_{legacy_ab}_{quantity}.txt",
    ]


def _etc_file_candidates(index: str) -> List[str]:
    idx = _to_state_label(index)
    num = _state_num(idx)
    legacy_ab = f"ab{num}"

    return [
        f"Analysis_{idx}_etc_ETC.txt",
        f"Analysis_{idx}_etc_Conductivity.txt",
        f"Analysis_{idx}_etc.txt",
        f"Analysis_{idx}_Conductivity.txt",
        f"Analysis_{idx}_ETC.txt",
        f"Analysis_{legacy_ab}_ETC.txt",
        f"Analysis_{legacy_ab}_Conductivity.txt",
    ]


def load_tm_state(index: str, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one state's TM results from txt: stiffness (4th-order) and CTE (3x3)."""
    stiff_candidates = [data_dir / name for name in _tm_file_candidates(index, "Stiffness")]
    cte_candidates = [data_dir / name for name in _tm_file_candidates(index, "CTE")]

    stiff_path = _first_existing_file(stiff_candidates)
    cte_path = _first_existing_file(cte_candidates)
    stiff_raw = _load_txt_matrix(stiff_path)
    cte_raw = _load_txt_matrix(cte_path)
    return _stiffness_from_input(stiff_raw), _cte_1x6_to_3x3(cte_raw)

def load_etc_state(index: str, data_dir: Path) -> np.ndarray:
    """Load one state's ETC result from txt as 3x3 symmetric tensor."""
    etc_candidates = [data_dir / name for name in _etc_file_candidates(index)]
    etc_path = _first_existing_file(etc_candidates)
    etc_raw = _load_txt_matrix(etc_path)
    return _symmetric_3x3_from_voigt6(etc_raw, name="ETC")


def _save_stiffness(index: str, c_v: np.ndarray, c_r: np.ndarray, out_dir: Path) -> None:
    base = f"Analysis_{index}_tm"
    np.savetxt(out_dir / f"{base}_Stiffness_V.txt", tu.tensor_voigt(c_v), fmt="%.6e", delimiter=",")
    np.savetxt(out_dir / f"{base}_Stiffness_R.txt", tu.tensor_voigt(c_r), fmt="%.6e", delimiter=",")


def _save_cte(index: str, alpha_v: np.ndarray, alpha_r: np.ndarray, out_dir: Path) -> None:
    base = f"Analysis_{index}_tm"
    np.savetxt(out_dir / f"{base}_CTE_V.txt", alpha_v, fmt="%.6e", delimiter=",")
    np.savetxt(out_dir / f"{base}_CTE_R.txt", alpha_r, fmt="%.6e", delimiter=",")


def _save_etc(index: str, k_v: np.ndarray, k_r: np.ndarray, out_dir: Path) -> None:
    base = f"Analysis_{index}_etc"
    np.savetxt(out_dir / f"{base}_ETC_V.txt", k_v, fmt="%.6e", delimiter=",")
    np.savetxt(out_dir / f"{base}_ETC_R.txt", k_r, fmt="%.6e", delimiter=",")


def _resolve_states(interp: str, indices: Sequence[str] | None) -> List[str]:
    required = LINEAR_NODE_IDS if interp == "linear" else STATE_ORDER

    if indices is None:
        return required

    states = [_to_state_label(s) for s in indices]

    if interp == "linear":
        if states != LINEAR_NODE_IDS:
            raise ValueError("Linear interpolation requires INDICES = ['b1', 'b2', 'b3'].")
        return states

    # quadratic
    if len(states) != 6:
        raise ValueError("Quadratic interpolation requires six indices: b1 b2 b3 b4 b5 b6.")
    if len(states) != len(set(states)):
        raise ValueError("Duplicate state labels in --indices are not allowed.")
    if set(states) != set(STATE_ORDER):
        raise ValueError("Quadratic interpolation requires full state set: b1 b2 b3 b4 b5 b6.")
    return STATE_ORDER


def _validate_analysis_types(raw_types: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    unsupported: List[str] = []

    for token in raw_types:
        t = token.lower()
        if t == "tm":
            normalized.extend(["elastic", "cte"])
        elif t in {"elastic", "stiffness"}:
            normalized.append("elastic")
        elif t == "cte":
            normalized.append("cte")
        elif t in {"etc", "conductivity", "thermal_conductivity"}:
            normalized.append("etc")
        else:
            unsupported.append(token)

    if unsupported:
        raise ValueError(
            f"Unsupported analysis type(s): {unsupported}. "
            "Expected any of ['elastic', 'cte', 'etc'] (or alias 'tm')."
        )

    # Keep order while removing duplicates.
    return list(dict.fromkeys(normalized))


def _run_linear(
    pn: np.ndarray,
    eig_vecs: np.ndarray,
    lin_ids: Sequence[str],
    state_tm: Dict[str, Tuple[np.ndarray, np.ndarray]],
    state_etc: Dict[str, np.ndarray],
    output_index: str,
    out_dir: Path,
    types: Sequence[str],
) -> None:
    weights = eu.find_coefficients(pn)

    print(f"[linear] states={lin_ids}, weights={weights}")

    if ("elastic" in types) or ("cte" in types):
        stiff_list = [state_tm[s][0] for s in lin_ids]
        cte_list = [state_tm[s][1] for s in lin_ids]
        beta_list = [tu.tensor_double_dot(c, a) for c, a in zip(stiff_list, cte_list)]

        c_v = weighted_sum(weights, stiff_list)
        beta_v = weighted_sum(weights, beta_list)
        c_v = tu.tensor_eigen_trans(c_v, eig_vecs)
        beta_v = tu.tensor_eigen_trans(beta_v, eig_vecs)
        a_v = tu.tensor_double_dot(tu.tensor_inverse(c_v), beta_v)

        s_list = [tu.tensor_inverse(c) for c in stiff_list]
        s_r = weighted_sum(weights, s_list)
        s_r = tu.tensor_eigen_trans(s_r, eig_vecs)
        c_r = tu.tensor_inverse(s_r)

        a_r = weighted_sum(weights, cte_list)
        a_r = tu.tensor_eigen_trans(a_r, eig_vecs)

        if "elastic" in types:
            _save_stiffness(output_index, c_v, c_r, out_dir)
        if "cte" in types:
            _save_cte(output_index, a_v, a_r, out_dir)

    if "etc" in types:
        k_list = [state_etc[s] for s in lin_ids]
        k_v = weighted_sum(weights, k_list)
        k_v = tu.tensor_eigen_trans(k_v, eig_vecs)
        k_r_inv = weighted_sum(weights, [np.linalg.inv(k) for k in k_list])
        k_r = np.linalg.inv(tu.tensor_eigen_trans(k_r_inv, eig_vecs))
        _save_etc(output_index, k_v, k_r, out_dir)


def _run_quadratic(
    pn: np.ndarray,
    eig_vecs: np.ndarray,
    states: Sequence[str],
    state_tm: Dict[str, Tuple[np.ndarray, np.ndarray]],
    state_etc: Dict[str, np.ndarray],
    output_index: str,
    out_dir: Path,
    types: Sequence[str],
) -> None:
    quad_ids = STATE_ORDER
    if set(states) != set(STATE_ORDER):
        raise ValueError("Quadratic interpolation requires all six states: b1 b2 b3 b4 b5 b6")

    weights = qi.t6_interpolate(pn, basis="bernstein")
    print(f"[quadratic] states={quad_ids}, weights={weights}")

    if ("elastic" in types) or ("cte" in types):
        stiff_list = [state_tm[s][0] for s in quad_ids]
        cte_list = [state_tm[s][1] for s in quad_ids]
        beta_list = [tu.tensor_double_dot(c, a) for c, a in zip(stiff_list, cte_list)]

        c_v = weighted_sum(weights, stiff_list)
        beta_v = weighted_sum(weights, beta_list)
        c_v = tu.tensor_eigen_trans(c_v, eig_vecs)
        beta_v = tu.tensor_eigen_trans(beta_v, eig_vecs)
        a_v = tu.tensor_double_dot(tu.tensor_inverse(c_v), beta_v)

        s_list = [tu.tensor_inverse(c) for c in stiff_list]
        s_r = weighted_sum(weights, s_list)
        s_r = tu.tensor_eigen_trans(s_r, eig_vecs)
        c_r = tu.tensor_inverse(s_r)

        a_r = weighted_sum(weights, cte_list)
        a_r = tu.tensor_eigen_trans(a_r, eig_vecs)

        if "elastic" in types:
            _save_stiffness(output_index, c_v, c_r, out_dir)
        if "cte" in types:
            _save_cte(output_index, a_v, a_r, out_dir)

    if "etc" in types:
        k_list = [state_etc[s] for s in quad_ids]
        k_v = weighted_sum(weights, k_list)
        k_v = tu.tensor_eigen_trans(k_v, eig_vecs)
        k_r_inv = weighted_sum(weights, [np.linalg.inv(k) for k in k_list])
        k_r = np.linalg.inv(tu.tensor_eigen_trans(k_r_inv, eig_vecs))
        _save_etc(output_index, k_v, k_r, out_dir)


def main() -> None:
    states = _resolve_states(INTERP, INDICES)
    analysis_types = _validate_analysis_types(ANALYSIS_TYPES)

    if ORIENTATION_CASE not in (1, 2, 3):
        raise ValueError("ORIENTATION_CASE must be 1, 2, or 3.")
    ori_2 = ORIENTATION_CASES[ORIENTATION_CASE - 1]
    eig_vecs, eig_vals = eu.sorted_eigens(ori_2)
    pn = np.array([eig_vals[0, 0], eig_vals[1, 1]], dtype=float)

    data_dir = _resolve_data_dir(SOURCE, DATA_DIR_OVERRIDE)
    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_index = OUTPUT_INDEX if OUTPUT_INDEX is not None else INTERP

    print(f"source={SOURCE}")
    print(f"interp={INTERP}")
    print(f"output_index={output_index}")
    print(f"data_dir={data_dir}")
    print(f"out_dir={out_dir}")
    print(f"states={states}")
    print(f"analysis_types={analysis_types}")
    print(f"pn={pn}")

    tm_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    etc_dict: Dict[str, np.ndarray] = {}

    load_ids = LINEAR_NODE_IDS if INTERP == "linear" else STATE_ORDER

    if ("elastic" in analysis_types) or ("cte" in analysis_types):
        for sid in load_ids:
            tm_dict[sid] = load_tm_state(sid, data_dir)

    if "etc" in analysis_types:
        for sid in load_ids:
            etc_dict[sid] = load_etc_state(sid, data_dir)

    if INTERP == "linear":
        _run_linear(pn, eig_vecs, LINEAR_NODE_IDS, tm_dict, etc_dict, output_index, out_dir, analysis_types)
    else:
        _run_quadratic(pn, eig_vecs, states, tm_dict, etc_dict, output_index, out_dir, analysis_types)


if __name__ == "__main__":
    main()


__all__ = [
    "main",
    "load_tm_state",
    "load_etc_state",
]
