from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_DIGIMAT_EXEC = Path(r"C:\MSC.Software\Digimat\2023.1\DigimatMF\exec\digimatDriver.exe")
DEFAULT_DIGIMAT_MF_BAT = Path(r"C:\MSC.Software\Digimat\2023.1\DigimatMF\exec\DigimatMF.bat")
SCRIPT_DIR = Path(__file__).resolve().parent


# ============================= User Config =============================
# Run only one DAF file each time.
INDEX = "a1"
ANALYSIS_TYPE = ["tm", "etc"][0]
ANALYSIS_TPYE = ANALYSIS_TYPE  # backward-compatible alias

# If None: auto use Analysis_{INDEX}_{ANALYSIS_TYPE}.daf, then Analysis_{INDEX}.daf
DAF_FILE: str | None = None
TMP_DIR = f"tmp_{INDEX}_{ANALYSIS_TYPE}"
JOB_NAME = f"Analysis_{INDEX}_{ANALYSIS_TYPE}"

ANALYSIS_DIR = SCRIPT_DIR
DIGIMAT_EXEC = DEFAULT_DIGIMAT_EXEC
DIGIMAT_MF_BAT = DEFAULT_DIGIMAT_MF_BAT

# For DigimatMF.bat you may set "-runMFWorkflow"; for digimatDriver.exe keep None.
WORKFLOW_FLAG: str | None = None
TIMEOUT_SECONDS: float | None = None
DRY_RUN = False
RUN_IN_BACKGROUND = True
BACKGROUND_STARTUP_CHECK_SECONDS = 3.0
USE_MF_ENV_BOOTSTRAP = True
SUBMIT_AS_MAT = True


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
    return candidates[0].resolve()


def _analysis_block_name(analysis_type: str) -> str:
    return "Analysis1" if analysis_type == "tm" else "Analysis2"


def _normalize_workflow_flag(flag: str) -> str:
    value = flag.strip()
    if not value:
        raise ValueError("WORKFLOW_FLAG cannot be empty.")
    if not value.startswith("-"):
        value = f"-{value}"
    if value.lower() == "-runmfworkflow":
        return "-runMFWorkflow"
    return value


def _default_workflow_flag(digimat_exec: Path) -> str | None:
    # DigimatMF.bat: use MF workflow flag.
    # digimatDriver.exe: no explicit workflow flag by default.
    name = digimat_exec.name.lower()
    if name.endswith(".bat"):
        return "-runMFWorkflow"
    return None


def _build_mf_env(
    digimat_exec: Path,
    digimat_mf_bat: Path | None = None,
    use_bootstrap: bool = True,
) -> dict[str, str]:
    env = os.environ.copy()
    if not use_bootstrap:
        return env

    # Infer install layout:
    # .../DigimatMF/exec/digimatDriver.exe
    # .../Digimat/exec
    exec_dir = digimat_exec.parent
    if digimat_mf_bat is not None:
        exec_dir = Path(digimat_mf_bat).parent
    release_root = exec_dir.parent.parent
    digimat_exec_dir = release_root / "Digimat" / "exec"
    digimat_fonts_dir = release_root / "Digimat" / "lib64" / "fonts"

    # Match the variables printed by DigimatMF.bat logs.
    env["DIGIMAT_BIN_20231"] = str(digimat_exec_dir)
    env["DIGIMAT_FONT_CACHE"] = str(digimat_fonts_dir)

    # Prepend Digimat folders to PATH so dependent DLLs are found.
    path_entries = [str(exec_dir), str(digimat_exec_dir), str(digimat_exec_dir.parent / "lib64")]
    old_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_entries + ([old_path] if old_path else []))
    return env


def _resolve_workflow_flag(digimat_exec: Path, workflow_flag: str | None) -> str | None:
    if workflow_flag is not None:
        return _normalize_workflow_flag(workflow_flag)
    return _default_workflow_flag(digimat_exec)


def _candidate_launchers(primary_exec: Path) -> list[Path]:
    parent = primary_exec.parent
    candidates = [
        primary_exec,
        parent / "digimat.exe",
        parent / "digimatDriver.exe",
        parent / "DigimatMF.bat",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            unique.append(p)
    return unique


def build_digimat_command(
    digimat_exec: Path,
    daf_path: Path,
    working_dir: Path | None = None,
    job_name: str | None = None,
    workflow_flag: str | None = None,
    arg_style: str = "kv",
) -> list[str]:
    if working_dir is None:
        working_dir = daf_path.parent
    if job_name is None:
        job_name = daf_path.stem

    cmd = [str(digimat_exec)]
    if workflow_flag:
        cmd.append(workflow_flag)

    if arg_style == "kv":
        cmd.extend(
            [
                f"input={daf_path}",
                f"workingDir={working_dir}",
                f"jobName={job_name}",
            ]
        )
    elif arg_style == "dash_space":
        cmd.extend(
            [
                "-input", str(daf_path),
                "-workingDir", str(working_dir),
                "-jobName", str(job_name),
            ]
        )
    elif arg_style == "dash_eq":
        cmd.extend(
            [
                f"-input={daf_path}",
                f"-workingDir={working_dir}",
                f"-jobName={job_name}",
            ]
        )
    elif arg_style == "slash_space":
        cmd.extend(
            [
                "/input", str(daf_path),
                "/workingDir", str(working_dir),
                "/jobName", str(job_name),
            ]
        )
    elif arg_style == "slash_eq":
        cmd.extend(
            [
                f"/input={daf_path}",
                f"/workingDir={working_dir}",
                f"/jobName={job_name}",
            ]
        )
    else:
        raise ValueError(f"Unsupported arg_style: {arg_style}")
    return cmd


def run_digimat_daf(
    daf_path: Path,
    digimat_exec: Path = DEFAULT_DIGIMAT_EXEC,
    digimat_mf_bat: Path | None = DEFAULT_DIGIMAT_MF_BAT,
    working_dir: Path | None = None,
    job_name: str | None = None,
    workflow_flag: str | None = None,
    timeout: float | None = None,
    dry_run: bool = False,
    run_in_background: bool = False,
    expected_output_path: Path | None = None,
    use_mf_env_bootstrap: bool = True,
) -> subprocess.CompletedProcess[str] | subprocess.Popen[str] | None:
    if not daf_path.exists():
        raise FileNotFoundError(f"DAF file not found: {daf_path}")
    if not dry_run and not digimat_exec.exists():
        raise FileNotFoundError(f"Digimat launcher not found: {digimat_exec}")

    launchers = _candidate_launchers(digimat_exec)
    if not launchers:
        raise FileNotFoundError(f"No Digimat launchers found near: {digimat_exec.parent}")

    env = _build_mf_env(
        digimat_exec=launchers[0],
        digimat_mf_bat=digimat_mf_bat,
        use_bootstrap=use_mf_env_bootstrap,
    )
    commands: list[tuple[Path, str, str | None, list[str]]] = []
    for launcher in launchers:
        resolved_workflow_flag = _resolve_workflow_flag(launcher, workflow_flag)
        workflow_candidates: list[str | None] = [resolved_workflow_flag]
        if resolved_workflow_flag is None:
            workflow_candidates.append("-runMFWorkflow")
        else:
            workflow_candidates.append(None)
        dedup_workflow: list[str | None] = []
        for wf in workflow_candidates:
            if wf not in dedup_workflow:
                dedup_workflow.append(wf)

        exec_name = launcher.name.lower()
        if exec_name.endswith(".exe"):
            arg_styles = ["kv", "dash_space", "dash_eq", "slash_space", "slash_eq"]
        else:
            arg_styles = ["kv"]

        for wf in dedup_workflow:
            for style in arg_styles:
                commands.append(
                    (
                        launcher,
                        style,
                        wf,
                        build_digimat_command(
                            digimat_exec=launcher,
                            daf_path=daf_path,
                            working_dir=working_dir,
                            job_name=job_name,
                            workflow_flag=wf,
                            arg_style=style,
                        ),
                    )
                )

    print("Executing (attempt 1):", " ".join(commands[0][3]))
    if dry_run:
        return None

    if run_in_background:
        if working_dir is None:
            working_dir = daf_path.parent
        if job_name is None:
            job_name = daf_path.stem

        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        failures: list[str] = []

        for attempt_idx, (launcher, style, wf, cmd) in enumerate(commands, start=1):
            wf_tag = "nowf" if wf is None else wf.lstrip("-")
            exe_tag = launcher.stem
            log_name = f"{job_name}.log" if attempt_idx == 1 else f"{job_name}.{exe_tag}.{wf_tag}.{style}.log"
            log_path = Path(working_dir) / log_name
            print(f"Attempt {attempt_idx}/{len(commands)} ({exe_tag},{wf_tag},{style}): {' '.join(cmd)}")
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                cwd=str(working_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                creationflags=creationflags,
            )
            log_file.close()
            time.sleep(max(0.0, float(BACKGROUND_STARTUP_CHECK_SECONDS)))

            return_code = process.poll()
            if return_code is None or (expected_output_path is not None and expected_output_path.exists()):
                print(f"Started in background, PID={process.pid}")
                print(f"Background log: {log_path}")
                return process

            preview = ""
            if log_path.exists():
                preview = log_path.read_text(encoding="utf-8", errors="ignore")[:400].strip()
            failures.append(
                f"{exe_tag}/{wf_tag}/{style}: rc={return_code}, log={log_path}, preview={preview if preview else '<empty>'}"
            )

        raise RuntimeError(
            "DigimatMF process exited immediately for all command variants. "
            + " | ".join(failures)
        )

    # Foreground mode: try command variants sequentially.
    last_exc: Exception | None = None
    for attempt_idx, (launcher, style, wf, cmd) in enumerate(commands, start=1):
        wf_tag = "nowf" if wf is None else wf.lstrip("-")
        exe_tag = launcher.stem
        print(f"Attempt {attempt_idx}/{len(commands)} ({exe_tag},{wf_tag},{style}): {' '.join(cmd)}")
        try:
            return subprocess.run(cmd, check=True, text=True, timeout=timeout, env=env)
        except Exception as exc:
            last_exc = exc
            continue

    assert last_exc is not None
    raise last_exc


def run_digimat_by_index(
    index: str = INDEX,
    analysis_type: str = ANALYSIS_TYPE,
    analysis_dir: Path = ANALYSIS_DIR,
    digimat_exec: Path = DEFAULT_DIGIMAT_EXEC,
    digimat_mf_bat: Path | None = DEFAULT_DIGIMAT_MF_BAT,
    daf_file: str | None = DAF_FILE,
    tmp_dir: str | None = None,
    job_name: str | None = None,
    workflow_flag: str | None = WORKFLOW_FLAG,
    timeout: float | None = None,
    dry_run: bool = False,
    run_in_background: bool = False,
    use_mf_env_bootstrap: bool = USE_MF_ENV_BOOTSTRAP,
) -> subprocess.CompletedProcess[str] | subprocess.Popen[str] | None:
    analysis_type = str(analysis_type).strip().lower()
    if analysis_type not in {"tm", "etc"}:
        raise ValueError(f"Unsupported ANALYSIS_TYPE: {analysis_type}. Expected 'tm' or 'etc'.")

    resolved_dir = _resolve_analysis_dir(analysis_dir)
    daf_path = _resolve_daf_path(resolved_dir, index=index, analysis_type=analysis_type, daf_file=daf_file)
    if not daf_path.exists():
        raise FileNotFoundError(f"DAF file not found: {daf_path}")

    tmp_dir_name = tmp_dir if tmp_dir else f"tmp_{index}_{analysis_type}"
    resolved_tmp_dir = resolved_dir / tmp_dir_name
    resolved_tmp_dir.mkdir(parents=True, exist_ok=True)

    resolved_job_name = job_name if job_name else f"Analysis_{index}_{analysis_type}"
    expected_eng_path = resolved_tmp_dir / f"{resolved_job_name}_{_analysis_block_name(analysis_type)}.eng"
    print(f"Expected ENG output: {expected_eng_path}")

    submit_input_path = daf_path
    if SUBMIT_AS_MAT and daf_path.suffix.lower() != ".mat":
        submit_input_path = resolved_tmp_dir / f"{Path(daf_path).stem}.mat"
        shutil.copy2(daf_path, submit_input_path)
        print(f"Submit input converted to MAT: {submit_input_path}")

    return run_digimat_daf(
        daf_path=submit_input_path,
        digimat_exec=digimat_exec,
        digimat_mf_bat=digimat_mf_bat,
        working_dir=resolved_tmp_dir,
        job_name=resolved_job_name,
        workflow_flag=workflow_flag,
        timeout=timeout,
        dry_run=dry_run,
        run_in_background=run_in_background,
        expected_output_path=expected_eng_path,
        use_mf_env_bootstrap=use_mf_env_bootstrap,
    )


def main() -> None:
    analysis_type = ANALYSIS_TPYE if ANALYSIS_TPYE else ANALYSIS_TYPE
    print(f"=== Running DigimatMF DAF ({analysis_type}) for index {INDEX} ===")
    run_digimat_by_index(
        index=INDEX,
        analysis_type=analysis_type,
        analysis_dir=ANALYSIS_DIR,
        digimat_exec=DIGIMAT_EXEC,
        digimat_mf_bat=DIGIMAT_MF_BAT,
        daf_file=DAF_FILE,
        tmp_dir=TMP_DIR,
        job_name=JOB_NAME,
        workflow_flag=WORKFLOW_FLAG,
        timeout=TIMEOUT_SECONDS,
        dry_run=DRY_RUN,
        run_in_background=RUN_IN_BACKGROUND,
        use_mf_env_bootstrap=USE_MF_ENV_BOOTSTRAP,
    )


if __name__ == "__main__":
    main()
