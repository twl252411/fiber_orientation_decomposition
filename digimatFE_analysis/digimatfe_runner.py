from __future__ import annotations

import subprocess
from pathlib import Path


DEFAULT_DIGIMAT_BAT = Path(r"C:\MSC.Software\Digimat\2023.1\DigimatFE\exec\DigimatFE.bat")

# ============================= User Config =============================
# Run only one DAF file each time.
INDEX = 0
ANALYSIS_DIR = Path("H:\github\fiber_orientation_decomposition\digimatFE_analysis")
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


def run_digimat_by_index(
    index: int,
    analysis_dir: Path = Path("digimatFE_analysis"),
    digimat_bat: Path = DEFAULT_DIGIMAT_BAT,
    use_run_fe_workflow_flag: bool = True,
    fallback_without_run_fe_workflow_flag: bool = True,
    timeout: float | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    daf_path = analysis_dir / f"Analysis_a{index}.daf"
    return run_digimat_daf(
        daf_path=daf_path,
        digimat_bat=digimat_bat,
        working_dir=analysis_dir,
        job_name=f"Analysis_a{index}",
        use_run_fe_workflow_flag=use_run_fe_workflow_flag,
        fallback_without_run_fe_workflow_flag=fallback_without_run_fe_workflow_flag,
        timeout=timeout,
        dry_run=dry_run,
    )


def main() -> None:
    print(f"=== Running Analysis_a{INDEX}.daf ===")
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
