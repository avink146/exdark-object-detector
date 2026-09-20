"""
Top-to-bottom Notebook Execution Engine.
Executes each cell in 03_baseline_training.ipynb sequentially,
captures real stdout, stderr, and matplotlib images,
and saves the authentic executed notebook.
"""

import sys
import os
import io
import json
import base64
import traceback
import time
from pathlib import Path

# Force UTF-8 and replace errors on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def safe_print(text, stream=sys.stdout):
    try:
        print(text, file=stream, flush=True)
    except Exception:
        try:
            safe_str = str(text).encode("ascii", errors="replace").decode("ascii")
            print(safe_str, file=stream, flush=True)
        except Exception:
            pass

def execute_notebook(nb_path: str = "notebooks/03_baseline_training.ipynb"):
    p = Path(nb_path).resolve()
    safe_print(f"Loading notebook: {p}")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Prepare persistent execution environment (shared namespace)
    global_ns = {
        "__name__": "__main__",
        "__file__": str(p),
    }

    execution_counter = 1
    total_code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    safe_print(f"Total code cells to execute: {total_code_cells}")

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        source_code = "".join(cell["source"])
        safe_print(f"\n[{execution_counter}/{total_code_cells}] Executing code cell...")

        cell["execution_count"] = execution_counter
        cell["outputs"] = []

        # Redirect stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        plt.close('all')
        start_time = time.time()
        err = None

        try:
            # Execute code in persistent namespace
            exec(source_code, global_ns)
        except Exception as e:
            err = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        out_text = stdout_capture.getvalue()
        err_text = stderr_capture.getvalue()

        # Echo to console safely
        if out_text:
            safe_print(out_text.strip())
            cell["outputs"].append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in out_text.splitlines()]
            })

        if err_text:
            safe_print(f"STDERR: {err_text.strip()}", stream=sys.stderr)
            cell["outputs"].append({
                "name": "stderr",
                "output_type": "stream",
                "text": [line + "\n" for line in err_text.splitlines()]
            })

        if err:
            safe_print(f"ERROR IN CELL {execution_counter}:\n{err}", stream=sys.stderr)
            cell["outputs"].append({
                "ename": "Exception",
                "evalue": str(err.splitlines()[-1]),
                "output_type": "error",
                "traceback": [line + "\n" for line in err.splitlines()]
            })
            # Save partial progress on failure
            with open(p, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            raise RuntimeError(f"Execution failed at cell {execution_counter}: {err}")

        # Check for open matplotlib figures to save as notebook display data
        fignums = plt.get_fignums()
        if fignums:
            for fig_id in fignums:
                fig = plt.figure(fig_id)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=120)
                img_buf.seek(0)
                img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")
                
                cell["outputs"].append({
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
            plt.close('all')

        execution_counter += 1

    # Save completely executed notebook
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    safe_print(f"\nAll {total_code_cells} cells executed successfully top-to-bottom!")
    safe_print(f"Saved executed notebook to: {p}")

if __name__ == "__main__":
    execute_notebook()
