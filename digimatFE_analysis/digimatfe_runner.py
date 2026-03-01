from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable


DEFAULT_DIGIMAT_BAT = Path(r"C:\MSC.Software\Digimat\2023.1\DigimatFE\exec\DigimatFE.bat")


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


def run_digimat_many(
    indices: Iterable[int],
    analysis_dir: Path = Path("digimatFE_analysis"),
    digimat_bat: Path = DEFAULT_DIGIMAT_BAT,
    use_run_fe_workflow_flag: bool = True,
    fallback_without_run_fe_workflow_flag: bool = True,
    timeout: float | None = None,
    dry_run: bool = False,
) -> None:
    for idx in indices:
        print(f"\n=== Running Analysis_a{idx}.daf ===")
        run_digimat_by_index(
            index=idx,
            analysis_dir=analysis_dir,
            digimat_bat=digimat_bat,
            use_run_fe_workflow_flag=use_run_fe_workflow_flag,
            fallback_without_run_fe_workflow_flag=fallback_without_run_fe_workflow_flag,
            timeout=timeout,
            dry_run=dry_run,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Digimat-FE .daf analyses with DigimatFE.bat."
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        required=True,
        help="Indices x for Analysis_a{x}.daf (for example: --indices 0 1 2).",
    )
    parser.add_argument("--analysis-dir", type=Path, default=Path("digimatFE_analysis"))
    parser.add_argument("--digimat-bat", type=Path, default=DEFAULT_DIGIMAT_BAT)
    parser.add_argument(
        "--no-runfe-flag",
        action="store_true",
        help="Do not pass -runFEWorkflow.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable retry without -runFEWorkflow.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds for each run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command only.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_digimat_many(
        indices=args.indices,
        analysis_dir=args.analysis_dir,
        digimat_bat=args.digimat_bat,
        use_run_fe_workflow_flag=not args.no_runfe_flag,
        fallback_without_run_fe_workflow_flag=not args.no_fallback,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
