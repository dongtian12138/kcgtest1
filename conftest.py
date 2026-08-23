"""Repository-wide pytest import roots; no runtime or simulator imports."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent
for relative in ("src/kcg_connector",):
    package_root = str(REPOSITORY_ROOT / relative)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
