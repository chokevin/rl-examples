"""Smoke-check that the local environment can run PufferLib examples."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile


PUFFERLIB_EXAMPLE_MODULES = (
    "pufferlib.emulation",
    "pufferlib.vector",
)

def check_example_modules() -> None:
    code = (
        "import importlib\n"
        "import shutil\n"
        "import gymnasium\n"
        "import pufferlib\n"
        f"modules = {PUFFERLIB_EXAMPLE_MODULES!r}\n"
        "for module_name in modules:\n"
        "    importlib.import_module(module_name)\n"
        "puffer_cli = shutil.which('puffer')\n"
        "if puffer_cli is None:\n"
        "    raise RuntimeError('Could not find the puffer CLI on PATH')\n"
        "print(f'pufferlib={getattr(pufferlib, \"__version__\", \"unknown\")}')\n"
        "print(f'gymnasium={gymnasium.__version__}')\n"
        "print(f'puffer={puffer_cli}')\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmpdir,
            text=True,
            capture_output=True,
            check=False,
        )

    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            [sys.executable, "-c", code],
            output=result.stdout,
            stderr=result.stderr,
        )

    if result.stdout:
        print(result.stdout, end="")


def main() -> None:
    puffer_cli = shutil.which("puffer")
    if puffer_cli is None:
        raise RuntimeError("Could not find the puffer CLI on PATH")
    check_example_modules()


if __name__ == "__main__":
    main()
