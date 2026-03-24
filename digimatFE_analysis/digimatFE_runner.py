from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


DEFAULT_DIGIMAT_BAT = Path(r"C:\\MSC.Software\\Digimat\\2023.1\\DigimatFE\\exec\\DigimatFE.bat")
SCRIPT_DIR = Path(__file__).resolve().parent

# ============================= User Config =============================
INDEXES = ["a2"]
ANALYSIS_TYPES = ["tm", "etc"]
# 批量选择：默认使用全部；可改成如 [INDEXES[0]] / [ANALYSIS_TYPES[0]]
SELECTED_INDEXES = INDEXES
SELECTED_ANALYSIS_TYPES = ANALYSIS_TYPES

ANALYSIS_DIR = SCRIPT_DIR
DIGIMAT_BAT = DEFAULT_DIGIMAT_BAT
USE_RUN_FE_WORKFLOW_FLAG = True
FALLBACK_WITHOUT_RUN_FE_WORKFLOW_FLAG = True
TIMEOUT_SECONDS: float | None = None
DRY_RUN = False


def _normalize_analysis_type(value: str) -> str:
    analysis_type = str(value).strip().lower()
    if analysis_type not in {"tm", "etc"}:
        raise ValueError(f"Unsupported ANALYSIS_TYPE: {value}. Expected 'tm' or 'etc'.")
    return analysis_type


def _normalize_indexes(indexes: Sequence[str]) -> list[str]:
    normalized = [str(idx).strip() for idx in indexes if str(idx).strip()]
    if not normalized:
        raise ValueError("SELECTED_INDEXES is empty.")
    return normalized


def _normalize_analysis_types(analysis_types: Sequence[str]) -> list[str]:
    normalized = [_normalize_analysis_type(v) for v in analysis_types]
    if not normalized:
        raise ValueError("SELECTED_ANALYSIS_TYPES is empty.")
    return normalized


def _resolve_analysis_dir(analysis_dir: Path) -> Path:
    analysis_dir = Path(analysis_dir)
    if analysis_dir.is_absolute():
        return analysis_dir.resolve()

    cwd_candidate = (Path.cwd() / analysis_dir).resolve()
    script_candidate = (SCRIPT_DIR / analysis_dir).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if script_candidate.exists():
        return script_candidate
    if analysis_dir.name == SCRIPT_DIR.name:
        return SCRIPT_DIR
    return script_candidate


def build_digimat_command(
    digimat_bat: Path,
    daf_path: Path,
    working_dir: Path | None = None,
    job_name: str | None = None,
    use_run_fe_workflow_flag: bool = True,
) -> list[str]:
    if working_dir is None:
        working_dir = daf_path.parent
    if job_name is None:
        job_name = daf_path.stem

    command = [str(digimat_bat)]
    if use_run_fe_workflow_flag:
        command.append("-runFEWorkflow")
    command.extend(
        [
            f"input={daf_path}",
            f"workingDir={working_dir}",
            f"jobName={job_name}",
        ]
    )
    return command


def run_digimat_daf(
    daf_path: Path,
    digimat_bat: Path = DEFAULT_DIGIMAT_BAT,
    working_dir: Path | None = None,
    job_name: str | None = None,
    use_run_fe_workflow_flag: bool = True,
    fallback_without_run_fe_workflow_flag: bool = True,
    timeout: float | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    if not daf_path.exists():
        raise FileNotFoundError(f"DAF file not found: {daf_path}")
    if not dry_run and not digimat_bat.exists():
        raise FileNotFoundError(f"DigimatFE.bat not found: {digimat_bat}")

    command = build_digimat_command(
        digimat_bat=digimat_bat,
        daf_path=daf_path,
        working_dir=working_dir,
        job_name=job_name,
        use_run_fe_workflow_flag=use_run_fe_workflow_flag,
    )
    print("Executing:", " ".join(command))
    if dry_run:
        return None

    try:
        return subprocess.run(command, check=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError:
        if use_run_fe_workflow_flag and fallback_without_run_fe_workflow_flag:
            fallback_command = build_digimat_command(
                digimat_bat=digimat_bat,
                daf_path=daf_path,
                working_dir=working_dir,
                job_name=job_name,
                use_run_fe_workflow_flag=False,
            )
            print("Retrying without -runFEWorkflow:", " ".join(fallback_command))
            return subprocess.run(fallback_command, check=True, text=True, timeout=timeout)
        raise


def _normalize_eng_name(working_dir: Path, job_name: str) -> Path | None:
    source_candidates = [
        working_dir / f"{job_name}_Analysis1.eng",
        working_dir / f"{job_name}_Analysis2.eng",
    ]
    target = working_dir / f"{job_name}.eng"
    for source in source_candidates:
        if source.exists():
            source.replace(target)
            print(f"Renamed ENG: {source.name} -> {target.name}")
            return target
    return None


def run_digimat_by_index(
    index: str,
    analysis_type: str,
    analysis_dir: Path = ANALYSIS_DIR,
    digimat_bat: Path = DEFAULT_DIGIMAT_BAT,
    use_run_fe_workflow_flag: bool = True,
    fallback_without_run_fe_workflow_flag: bool = True,
    timeout: float | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    analysis_type = _normalize_analysis_type(analysis_type)
    index = str(index).strip()
    if not index:
        raise ValueError("index is empty.")

    resolved_dir = _resolve_analysis_dir(analysis_dir)
    daf_path = (resolved_dir / f"Analysis_{index}_{analysis_type}.daf").resolve()
    tmp_dir = resolved_dir / f"tmp_{index}_{analysis_type}"
    job_name = f"Analysis_{index}_{analysis_type}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    result = run_digimat_daf(
        daf_path=daf_path,
        digimat_bat=digimat_bat,
        working_dir=tmp_dir,
        job_name=job_name,
        use_run_fe_workflow_flag=use_run_fe_workflow_flag,
        fallback_without_run_fe_workflow_flag=fallback_without_run_fe_workflow_flag,
        timeout=timeout,
        dry_run=dry_run,
    )
    if not dry_run:
        _normalize_eng_name(working_dir=tmp_dir, job_name=job_name)
    return result


def run_batch(
    indexes: Sequence[str] = SELECTED_INDEXES,
    analysis_types: Sequence[str] = SELECTED_ANALYSIS_TYPES,
    analysis_dir: Path = ANALYSIS_DIR,
    digimat_bat: Path = DIGIMAT_BAT,
    use_run_fe_workflow_flag: bool = USE_RUN_FE_WORKFLOW_FLAG,
    fallback_without_run_fe_workflow_flag: bool = FALLBACK_WITHOUT_RUN_FE_WORKFLOW_FLAG,
    timeout: float | None = TIMEOUT_SECONDS,
    dry_run: bool = DRY_RUN,
) -> list[subprocess.CompletedProcess[str] | None]:
    normalized_indexes = _normalize_indexes(indexes)
    normalized_types = _normalize_analysis_types(analysis_types)

    results: list[subprocess.CompletedProcess[str] | None] = []
    for idx in normalized_indexes:
        for a_type in normalized_types:
            print(f"=== Running DigimatFE input ({a_type}) for index {idx} ===")
            results.append(
                run_digimat_by_index(
                    index=idx,
                    analysis_type=a_type,
                    analysis_dir=analysis_dir,
                    digimat_bat=digimat_bat,
                    use_run_fe_workflow_flag=use_run_fe_workflow_flag,
                    fallback_without_run_fe_workflow_flag=fallback_without_run_fe_workflow_flag,
                    timeout=timeout,
                    dry_run=dry_run,
                )
            )
    return results


def main() -> None:
    run_batch()


if __name__ == "__main__":
    main()
