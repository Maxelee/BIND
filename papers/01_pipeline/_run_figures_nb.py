"""Execute paper1_figures.ipynb in place with the BIND venv kernel.

Pins the kernel to THIS interpreter (sys.executable) via kernel_cmd so a stray
conda kernelspec can never be picked up. Run from papers/01_pipeline/:
    /mnt/home/mlee1/venvs/BIND_env/bin/python _run_figures_nb.py
"""
import os
import sys
import time

import nbformat
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient

NB = "paper1_figures.ipynb"


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
