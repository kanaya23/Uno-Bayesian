"""Entry point for local UNO BNN training.

Provides a thin wrapper around the notebook-derived CLI while surfacing
dependency errors in a friendlier way for end users.
"""

from __future__ import annotations

import sys
from typing import NoReturn


def _bail(message: str, *, exit_code: int = 1) -> NoReturn:
    print(f"[train_local_bnn] {message}", file=sys.stderr)
    sys.exit(exit_code)


try:
    from notebooks.uno_bnn_curriculum_converted import run_cli
except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
    missing = exc.name or getattr(exc.__cause__, "name", None) or "a required dependency"
    if any(flag in sys.argv for flag in ("-h", "--help")):
        print(
            "UNO BNN trainer (lightweight help)\n"
            f"Dependencies not yet installed (missing: {missing}).\n"
            "Install core packages with:\n"
            "    pip install numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu pyro-ppl tqdm\n"
            "After installing, re-run `python train_local_bnn.py --help` for the full CLI details."
        )
        sys.exit(0)
    hint = (
        "pip install numpy torch torchvision torchaudio --index-url "
        "https://download.pytorch.org/whl/cpu pyro-ppl tqdm"
    )
    _bail(
        f"Missing dependency '{missing}'. Install required packages first:\n"
        f"    {hint}\nOriginal error: {exc}"
    )


if __name__ == "__main__":
    try:
        run_cli()
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        _bail(str(exc))
    except KeyboardInterrupt:  # pragma: no cover - signal handling
        _bail("Training interrupted by user.", exit_code=130)
