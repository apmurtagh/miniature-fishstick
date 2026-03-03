"""
Environment provenance capture.

Writes lightweight provenance files into ``<run_dir>/meta/``:

* ``python_version.txt``  – output of ``python --version``
* ``pip_freeze.txt``      – output of ``pip freeze``
* ``conda_history.txt``   – output of ``conda env export --from-history``
                            (only written when conda is available)
"""

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str:
    """Run *cmd* and return stdout; return empty string on any error."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout
    except Exception:
        return ""


def capture_env(meta_dir: Path) -> None:
    """Write provenance files into *meta_dir*.

    Parameters
    ----------
    meta_dir:
        Directory in which to write provenance files.  Created if absent.
    """
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Python version
    python_version = f"Python {sys.version}\n"
    (meta_dir / "python_version.txt").write_text(python_version, encoding="utf-8")

    # pip freeze
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])
    (meta_dir / "pip_freeze.txt").write_text(pip_freeze, encoding="utf-8")

    # conda env export --from-history (best-effort)
    conda_history = _run(["conda", "env", "export", "--from-history"])
    if conda_history.strip():
        (meta_dir / "conda_history.txt").write_text(conda_history, encoding="utf-8")
