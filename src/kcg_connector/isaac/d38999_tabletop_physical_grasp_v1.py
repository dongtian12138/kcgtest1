#!/usr/bin/env python3
"""Formal entry point for sensor-bounded D38999 tabletop grasping.

The physical scene and robot are intentionally reused from the validated
tabletop pick runner.  This file exists only to provide an unambiguous command
name; the implementation remains single-sourced.
"""

from d38999_tabletop_pick_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
