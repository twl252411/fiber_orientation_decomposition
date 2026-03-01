from __future__ import annotations

import subprocess
from pathlib import Path


DEFAULT_DIGIMAT_BAT = Path(r"C:\MSC.Software\Digimat\2023.1\DigimatMF\exec\DigimatMF.bat")
SCRIPT_DIR = Path(__file__).resolve().parent


# ============================= User Config =============================
# Run only one DAF file each time.
INDEX = "a1"
ANALYSIS_TPYE = ["tm", "etc"][0]
DAF_FILE: str | None = None  # e.g. "Analysis_a1_tm.daf"; None -> auto resolve by INDEX/ANALYSIS_TPYE
TMP_DIR = f"tmp_{INDEX}_{ANALYSIS_TPYE}"
JOB_NAME = f"Analysis_{INDEX}_{ANALYSIS_TPYE}"

ANALYSIS_DIR = SCRIPT_DIR
DIGIMAT_BAT = DEFAULT_DIGIMAT_BAT
USE_RUN_FE_WORKFLOW_FLAG = True
FALLBACK_WITHOUT_RUN_FE_WORKFLOW_FLAG = True
TIMEOUT_SECONDS: float | None = None
DRY_RUN = False


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


def _resolve_daf_path(
    analysis_dir: Path,
    index: str,
    analysis_type: str,
    daf_file: str | None,
) -> Path:
    candidates: list[Path] = []
    if daf_file:
        candidates.append(analysis_dir / daf_file)
    candidates.append(analysis_dir / f"Analysis_{index}_{analysis_type}.daf")
    candidates.append(analysis_dir / f"Analysis_{index}.daf")

    for path in candidates:
        if path.exists():
            return path.resolve()

    # Return the primary target path in the error message.
    return candidates[0].resolve()


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


def run_digimat_by_index(
    index: str = INDEX,
    analysis_type: str = ANALYSIS_TPYE,
    analysis_dir: Path = ANALYSIS_DIR,
    digimat_bat: Path = DEFAULT_DIGIMAT_BAT,
    daf_file: str | None = DAF_FILE,
    tmp_dir: str | None = None,
    job_name: str | None = None,
    use_run_fe_workflow_flag: bool = True,
    fallback_without_run_fe_workflow_flag: bool = True,
    timeout: float | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    analysis_type = str(analysis_type).strip().lower()
    if analysis_type not in {"tm", "etc"}:
        raise ValueError(f"Unsupported ANALYSIS_TPYE: {analysis_type}. Expected 'tm' or 'etc'.")

    resolved_dir = _resolve_analysis_dir(analysis_dir)
    daf_path = _resolve_daf_path(resolved_dir, index=index, analysis_type=analysis_type, daf_file=daf_file)
    if not daf_path.exists():
        raise FileNotFoundError(f"DAF file not found: {daf_path}")

    tmp_dir_name = tmp_dir if tmp_dir else f"tmp_{index}_{analysis_type}"
    resolved_tmp_dir = resolved_dir / tmp_dir_name
    resolved_tmp_dir.mkdir(parents=True, exist_ok=True)

    resolved_job_name = job_name if job_name else f"Analysis_{index}_{analysis_type}"
    return run_digimat_daf(
        daf_path=daf_path,
        digimat_bat=digimat_bat,
        working_dir=resolved_tmp_dir,
        job_name=resolved_job_name,
        use_run_fe_workflow_flag=use_run_fe_workflow_flag,
        fallback_without_run_fe_workflow_flag=fallback_without_run_fe_workflow_flag,
        timeout=timeout,
        dry_run=dry_run,
    )


def main() -> None:
    print(f"=== Running DAF ({ANALYSIS_TPYE}) for index {INDEX} ===")
    run_digimat_by_index(
        index=INDEX,
        analysis_type=ANALYSIS_TPYE,
        analysis_dir=ANALYSIS_DIR,
        digimat_bat=DIGIMAT_BAT,
        daf_file=DAF_FILE,
        tmp_dir=TMP_DIR,
        job_name=JOB_NAME,
        use_run_fe_workflow_flag=USE_RUN_FE_WORKFLOW_FLAG,
        fallback_without_run_fe_workflow_flag=FALLBACK_WITHOUT_RUN_FE_WORKFLOW_FLAG,
        timeout=TIMEOUT_SECONDS,
        dry_run=DRY_RUN,
    )


if __name__ == "__main__":
    main()
