"""Execute mechanism_check.ipynb in place with the BIND venv kernel.

Same pattern as _run_figures_nb.py (kernel pinned to THIS interpreter).
Run from papers/01_pipeline/:
    /mnt/home/mlee1/venvs/BIND_env/bin/python _run_mechanism_nb.py
"""
import os
import sys
import time

import nbformat
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient

NB = "mechanism_check.ipynb"


def main() -> None:
    nb = nbformat.read(NB, as_version=4)
    km = KernelManager()
    km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, km=km, timeout=int(os.environ.get("NB_CELL_TIMEOUT", "3600")),
                            resources={"metadata": {"path": "."}})
    t0 = time.time()
    try:
        client.execute()
    finally:
        nbformat.write(nb, NB)          # keep partial outputs on failure
    print(f"executed {NB} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
