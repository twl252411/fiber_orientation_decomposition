from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DIGIMAT_BATCH_BAT = Path(r"C:\MSC.Software\Digimat\2023.1\DigimatMF\exec\DigimatMF_batch.bat")
DEFAULT_DIGIMAT_NOGUI_BAT = Path(r"C:\MSC.Software\Digimat\2023.1\DigimatMF\exec\DigimatMF_nogui.bat")
DEFAULT_DIGIMAT_EXE = Path(r"C:\MSC.Software\Digimat\2023.1\DigimatMF\exec\digimat.exe")


# ============================= User Config =============================
INDEX = "a1"
ANALYSIS_TYPE = ["tm", "etc"][0]

INPUT_FILE: str | None = None
ANALYSIS_DIR = SCRIPT_DIR

TMP_DIR = f"tmp_{INDEX}_{ANALYSIS_TYPE}"
JOB_NAME = f"Analysis_{INDEX}_{ANALYSIS_TYPE}"

RUNNER_BACKEND = ["batch_bat", "digimat_exe"][0]
DIGIMAT_BATCH_BAT = DEFAULT_DIGIMAT_BATCH_BAT
DIGIMAT_NOGUI_BAT = DEFAULT_DIGIMAT_NOGUI_BAT
DIGIMAT_EXE = DEFAULT_DIGIMAT_EXE
ALLOW_GUI_FALLBACK = False

LICENSE_WAIT = False
RUN_IN_BACKGROUND = True
SUBMIT_AS_MAT = True
STAGE_INPUT_IN_TMP = True
TIMEOUT_SECONDS: float | None = None
DRY_RUN = False


def _sanitize_path_text(value: Path | str) -> str:
    text = str(value).strip()
    return text.strip('"').strip("'")


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
    return script_candidate


def _resolve_input_file(
    analysis_dir: Path,
    index: str,
    analysis_type: str,
    input_file: str | None,
) -> Path:
    candidates: list[Path] = []
    if input_file:
        candidates.append(analysis_dir / input_file)
    candidates.append(analysis_dir / f"Analysis_{index}_{analysis_type}.mat")
    candidates.append(analysis_dir / f"Analysis_{index}.mat")
    candidates.append(analysis_dir / f"Analysis_{index}_{analysis_type}.daf")
    candidates.append(analysis_dir / f"Analysis_{index}.daf")

    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def _analysis_block_name(analysis_type: str) -> str:
    return "Analysis1" if analysis_type == "tm" else "Analysis2"


def _resolve_batch_bat(batch_bat: Path, nogui_bat: Path, allow_gui_fallback: bool) -> Path:
    preferred = Path(_sanitize_path_text(batch_bat))
    preferred_nogui = Path(_sanitize_path_text(nogui_bat))
    candidates = [
        preferred_nogui,
        preferred,
        preferred.parent / "DigimatMF_nogui.bat",
        preferred.parent / "DigimatMF_batch.bat",
    ]
    if allow_gui_fallback:
        candidates.append(preferred.parent / "DigimatMF.bat")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return preferred


def _build_direct_exe_cmd(digimat_exe: Path, input_file: Path, license_wait: bool) -> list[str]:
    cmd = [str(digimat_exe), f"input={str(input_file.resolve())}"]
    if license_wait:
        cmd.append("-licensewait")
    return cmd


def _stage_submit_input(
    input_path: Path,
    tmp_dir: Path,
    job_name: str,
    submit_as_mat: bool,
    stage_input_in_tmp: bool,
) -> Path:
    if submit_as_mat:
        if input_path.suffix.lower() not in {".mat", ".daf"}:
            raise ValueError(f"Unsupported input suffix for MAT submission: {input_path.suffix}")
        staged_path = tmp_dir / f"{job_name}.mat"
    elif stage_input_in_tmp:
        staged_path = tmp_dir / f"{job_name}{input_path.suffix}"
    else:
        return input_path

    if input_path.resolve() != staged_path.resolve():
        shutil.copy2(input_path, staged_path)
        print(f"Staged input in tmp: {staged_path}")
    return staged_path


def _write_batch_wrapper(
    batch_bat: Path,
    input_file: Path,
    working_dir: Path,
    job_name: str | None,
    license_wait: bool,
) -> Path:
    wrapper_name = f"_run_{job_name if job_name else input_file.stem}.cmd"
    wrapper_path = working_dir / wrapper_name

    args = f'input="{_sanitize_path_text(input_file.resolve())}"'
    if license_wait:
        args += " -licensewait"

    script = (
        "@echo off\n"
        "setlocal\n"
        f'cd /d "{_sanitize_path_text(working_dir.resolve())}"\n'
        f'call "{_sanitize_path_text(batch_bat)}" {args}\n'
        "exit /b %errorlevel%\n"
    )
    wrapper_path.write_text(script, encoding="utf-8", newline="\n")
    return wrapper_path


def run_digimat_input(
    input_file: Path,
    backend: str = RUNNER_BACKEND,
    batch_bat: Path = DEFAULT_DIGIMAT_BATCH_BAT,
    nogui_bat: Path = DEFAULT_DIGIMAT_NOGUI_BAT,
    digimat_exe: Path = DEFAULT_DIGIMAT_EXE,
    working_dir: Path | None = None,
    license_wait: bool = False,
    run_in_background: bool = False,
    timeout: float | None = None,
    dry_run: bool = False,
    job_name: str | None = None,
    allow_gui_fallback: bool = ALLOW_GUI_FALLBACK,
) -> subprocess.CompletedProcess[str] | subprocess.Popen[str] | None:
    input_file = Path(input_file).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if working_dir is None:
        working_dir = input_file.parent
    working_dir = Path(working_dir).resolve()
    working_dir.mkdir(parents=True, exist_ok=True)

    backend = str(backend).strip().lower()
    if backend not in {"batch_bat", "digimat_exe"}:
        raise ValueError(f"Unsupported RUNNER_BACKEND: {backend}. Expected 'batch_bat' or 'digimat_exe'.")

    if backend == "batch_bat":
        batch_bat_path = _resolve_batch_bat(
            batch_bat=Path(batch_bat),
            nogui_bat=Path(nogui_bat),
            allow_gui_fallback=allow_gui_fallback,
        )
        if not dry_run and not batch_bat_path.exists():
            raise FileNotFoundError(f"Digimat batch bat not found: {batch_bat_path}")
        if not allow_gui_fallback and batch_bat_path.name.lower() == "digimatmf.bat":
            raise RuntimeError(
                f"Refusing GUI launcher in no-GUI mode: {batch_bat_path}. "
                "Set ALLOW_GUI_FALLBACK=True if you really want GUI fallback."
            )
        print(f"Using batch launcher: {batch_bat_path}")
        wrapper = _write_batch_wrapper(
            batch_bat=batch_bat_path,
            input_file=input_file,
            working_dir=working_dir,
            job_name=job_name,
            license_wait=license_wait,
        )
        cmd: list[str] = ["cmd", "/d", "/c", str(wrapper)]
    else:
        digimat_exe_path = Path(_sanitize_path_text(digimat_exe))
        if not dry_run and not digimat_exe_path.exists():
            raise FileNotFoundError(f"digimat.exe not found: {digimat_exe_path}")
        cmd = _build_direct_exe_cmd(digimat_exe=digimat_exe_path, input_file=input_file, license_wait=license_wait)

    print("Executing:", subprocess.list2cmdline(cmd))
    if dry_run:
        return None

    if run_in_background:
        log_name = f"{job_name}.log" if job_name else f"{input_file.stem}.log"
        log_path = working_dir / log_name
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(working_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        log_file.close()
        print(f"Started in background, PID={process.pid}")
        print(f"Background log: {log_path}")
        return process

    result = subprocess.run(
        cmd,
        cwd=str(working_dir),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    print(f"Return code: {result.returncode}")
    if result.stdout.strip():
        print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)
    return result


def run_digimat_by_index(
    index: str = INDEX,
    analysis_type: str = ANALYSIS_TYPE,
    analysis_dir: Path = ANALYSIS_DIR,
    input_file: str | None = INPUT_FILE,
    tmp_dir: str | None = None,
    job_name: str | None = None,
    backend: str = RUNNER_BACKEND,
    batch_bat: Path = DIGIMAT_BATCH_BAT,
    nogui_bat: Path = DIGIMAT_NOGUI_BAT,
    digimat_exe: Path = DIGIMAT_EXE,
    allow_gui_fallback: bool = ALLOW_GUI_FALLBACK,
    license_wait: bool = LICENSE_WAIT,
    submit_as_mat: bool = SUBMIT_AS_MAT,
    stage_input_in_tmp: bool = STAGE_INPUT_IN_TMP,
    timeout: float | None = None,
    dry_run: bool = False,
    run_in_background: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.Popen[str] | None:
    analysis_type = str(analysis_type).strip().lower()
    if analysis_type not in {"tm", "etc"}:
        raise ValueError(f"Unsupported ANALYSIS_TYPE: {analysis_type}. Expected 'tm' or 'etc'.")

    resolved_dir = _resolve_analysis_dir(analysis_dir)
    input_path = _resolve_input_file(resolved_dir, index=index, analysis_type=analysis_type, input_file=input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    tmp_dir_name = tmp_dir if tmp_dir else f"tmp_{index}_{analysis_type}"
    resolved_tmp_dir = resolved_dir / tmp_dir_name
    resolved_tmp_dir.mkdir(parents=True, exist_ok=True)

    resolved_job_name = job_name if job_name else f"Analysis_{index}_{analysis_type}"
    expected_eng_path = resolved_tmp_dir / f"{resolved_job_name}_{_analysis_block_name(analysis_type)}.eng"
    print(f"Expected ENG output: {expected_eng_path}")

    submit_input = _stage_submit_input(
        input_path=input_path,
        tmp_dir=resolved_tmp_dir,
        job_name=resolved_job_name,
        submit_as_mat=submit_as_mat,
        stage_input_in_tmp=stage_input_in_tmp,
    )

    return run_digimat_input(
        input_file=submit_input,
        backend=backend,
        batch_bat=batch_bat,
        nogui_bat=nogui_bat,
        digimat_exe=digimat_exe,
        working_dir=resolved_tmp_dir,
        license_wait=license_wait,
        run_in_background=run_in_background,
        timeout=timeout,
        dry_run=dry_run,
        job_name=resolved_job_name,
        allow_gui_fallback=allow_gui_fallback,
    )


# Backward-compatible alias
def run_via_bat(
    input_file: Path,
    digimat_bat: Path = DIGIMAT_BATCH_BAT,
    nogui_bat: Path = DIGIMAT_NOGUI_BAT,
    working_dir: Path | None = None,
    allow_gui_fallback: bool = ALLOW_GUI_FALLBACK,
    license_wait: bool = LICENSE_WAIT,
    run_in_background: bool = False,
    timeout: float | None = None,
    dry_run: bool = False,
    job_name: str | None = None,
) -> subprocess.CompletedProcess[str] | subprocess.Popen[str] | None:
    return run_digimat_input(
        input_file=input_file,
        backend="batch_bat",
        batch_bat=digimat_bat,
        nogui_bat=nogui_bat,
        digimat_exe=DIGIMAT_EXE,
        working_dir=working_dir,
        allow_gui_fallback=allow_gui_fallback,
        license_wait=license_wait,
        run_in_background=run_in_background,
        timeout=timeout,
        dry_run=dry_run,
        job_name=job_name,
    )


def main() -> None:
    print(f"=== Running DigimatMF input ({ANALYSIS_TYPE}) for index {INDEX} ===")
    run_digimat_by_index(
        index=INDEX,
        analysis_type=ANALYSIS_TYPE,
        analysis_dir=ANALYSIS_DIR,
        input_file=INPUT_FILE,
        tmp_dir=TMP_DIR,
        job_name=JOB_NAME,
        backend=RUNNER_BACKEND,
        batch_bat=DIGIMAT_BATCH_BAT,
        nogui_bat=DIGIMAT_NOGUI_BAT,
        digimat_exe=DIGIMAT_EXE,
        allow_gui_fallback=ALLOW_GUI_FALLBACK,
        license_wait=LICENSE_WAIT,
        submit_as_mat=SUBMIT_AS_MAT,
        stage_input_in_tmp=STAGE_INPUT_IN_TMP,
        timeout=TIMEOUT_SECONDS,
        dry_run=DRY_RUN,
        run_in_background=RUN_IN_BACKGROUND,
    )


if __name__ == "__main__":
    main()
