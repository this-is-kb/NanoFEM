"""Phase-0 smoke example: the package imports and states its version.

Real examples arrive with the walking skeleton (phase 0.5).
"""

from __future__ import annotations

import nanofem


def main() -> None:
    """Print the installed NanoFEM version."""
    print(f"NanoFEM {nanofem.__version__} - phase 0 architectural skeleton")


if __name__ == "__main__":
    main()
