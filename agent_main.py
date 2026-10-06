"""Entry point for the packaged agent.

PyInstaller needs a plain script at the project root: `python -m app.agent`
is not something it can package, because there is no module execution context
inside a frozen build.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()   # Windows: stops a frozen exe re-running itself
    from app.agent import main
    sys.exit(main())
