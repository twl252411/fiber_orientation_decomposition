from __future__ import annotations

import subprocess
from pathlib import Path


DEFAULT_DIGIMAT_BAT = Path(r"C:\\MSC.Software\\Digimat\\2023.1\\DigimatFE\\exec\\DigimatFE.bat")
SCRIPT_DIR = Path(__file__).resolve().parent

# ============================= User Config =============================
# Run only one DAF file each time.
INDEX = "a2"
ANALYSIS_TYPE = ["tm", "etc"][1]
DAF_FILE = f"Analysis_{INDEX}_{ANALYSIS_TYPE}.daf"
TMP_DIR = f"tmp_{INDEX}_{ANALYSIS_TYPE}"
JAB_NAME = f"Analysis_{INDEX}_{ANALYSIS_TYPE}"

ANALYSIS_DIR = SCRIPT_DIR
DIGIMAT_BAT = DEFAULT_DIGIMAT_BAT
USE_RUN_FE_WORKFLOW_FLAG = True
FALLBACK_WITHOUT_RUN_FE_WORKFLOW_FLAG = True
TIMEOUT_SECONDS: float | None = None
DRY_RUN = False


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
    # Digimat often writes {job}_Analysis1.eng (or Analysis2). Normalize to {job}.eng.
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
    index: str | int,
    analysis_dir: Path = ANALYSIS_DIR,
    digimat_bat: Path = DEFAULT_DIGIMAT_BAT,
    use_run_fe_workflow_flag: bool = True,
    fallback_without_run_fe_workflow_flag: bool = True,
    timeout: float | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.is_absolute():
        cwd_candidate = (Path.cwd() / analysis_dir).resolve()
        script_candidate = (SCRIPT_DIR / analysis_dir).resolve()
        if cwd_candidate.exists():
            analysis_dir = cwd_candidate
        elif script_candidate.exists():
            analysis_dir = script_candidate
        elif analysis_dir.name == SCRIPT_DIR.name:
            # Common misconfiguration: analysis_dir="digimatFE_analysis" while script already lives there.
            analysis_dir = SCRIPT_DIR
        else:
            analysis_dir = script_candidate
    else:
        analysis_dir = analysis_dir.resolve()

    daf_path = (analysis_dir / DAF_FILE).resolve()
    tmp_dir = analysis_dir / TMP_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)

    result = run_digimat_daf(
        daf_path=daf_path,
        digimat_bat=digimat_bat,
        working_dir=tmp_dir,
        job_name=JAB_NAME,
        use_run_fe_workflow_flag=use_run_fe_workflow_flag,
        fallback_without_run_fe_workflow_flag=fallback_without_run_fe_workflow_flag,
        timeout=timeout,
        dry_run=dry_run,
    )
    if not dry_run:
        _normalize_eng_name(working_dir=tmp_dir, job_name=JAB_NAME)
    return result


def main() -> None:
    print(f"=== Running Analysis.daf ===")
    run_digimat_by_index(
        index=INDEX,
        analysis_dir=ANALYSIS_DIR,
        digimat_bat=DIGIMAT_BAT,
        use_run_fe_workflow_flag=USE_RUN_FE_WORKFLOW_FLAG,
        fallback_without_run_fe_workflow_flag=FALLBACK_WITHOUT_RUN_FE_WORKFLOW_FLAG,
        timeout=TIMEOUT_SECONDS,
        dry_run=DRY_RUN,
    )


if __name__ == "__main__":
    main()
