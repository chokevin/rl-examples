"""Smoke-check that the local environment can run PufferLib examples."""

from __future__ import annotations

import subprocess
import sys
import tempfile


PUFFERLIB_EXAMPLE_MODULES = ("pufferlib.models",)

def check_example_modules() -> None:
    code = (
        "import importlib\n"
        "import importlib.metadata\n"
        "import shutil\n"
        "import pufferlib\n"
        "import gymnasium\n"
        f"modules = {PUFFERLIB_EXAMPLE_MODULES!r}\n"
        "for module_name in modules:\n"
        "    importlib.import_module(module_name)\n"
        "puffer_cli = shutil.which('puffer') or '<not installed>'\n"
        "print(f'pufferlib={getattr(pufferlib, \"__version__\", \"unknown\")}')\n"
        "print(f'gymnasium={importlib.metadata.version(\"gymnasium\")}')\n"
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
    if result.stderr:
        sys.stderr.write(result.stderr)


def main() -> None:
    check_example_modules()


if __name__ == "__main__":
    main()
