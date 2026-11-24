"""Compatibility wrapper for the v4 training entry point.

Use ``python -m members.member04_hog.final_v4.train --method ae|supervised``.
"""
from .final_v4.train import main

if __name__ == "__main__":
    main()
