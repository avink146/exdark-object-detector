"""
Cell-by-Cell Jupyter Kernel Execution Engine.
Uses a real IPython kernel via nbclient to execute each code cell,
inspect its outputs, validate execution, and save the notebook with real embedded outputs.
"""

import sys
import os
import time
import asyncio
from pathlib import Path
import nbformat
from nbclient import NotebookClient

# Fix Windows asyncio loop for zmq
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def safe_log(msg):
    try:
        print(msg, flush=True)
    except Exception:
        try:
            print(str(msg).encode("ascii", errors="replace").decode("ascii"), flush=True)
        except Exception:
            pass

def execute_notebook_with_kernel(nb_path: str, timeout: int = 1200):
    p = Path(nb_path).resolve()
    safe_log(f"\n{'='*70}")
    safe_log(f"LAUNCHING REAL JUPYTER KERNEL FOR: {p.name}")
    safe_log(f"Full Path: {p}")
    safe_log(f"{'='*70}")

    with open(p, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    total_cells = len(nb.cells)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    safe_log(f"Total cells: {total_cells} | Code cells: {len(code_cells)}")

    # Initialize client with current workspace as working directory
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(p.parent.parent.resolve())}}
    )

    code_idx = 0
    start_all = time.time()

    # Use client setup and teardown to execute cell-by-cell
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                code_idx += 1
                safe_log(f"\n[{code_idx}/{len(code_cells)}] Executing Cell #{i+1}...")
                
                # Show first 3 lines of code
                src_lines = cell.source.strip().splitlines()
                preview = " | ".join(src_lines[:2])
                if len(src_lines) > 2:
                    preview += " ..."
                safe_log(f"  Code: {preview}")

                t0 = time.time()
                try:
                    client.execute_cell(cell, i, execution_count=code_idx)
                except Exception as e:
                    safe_log(f"  ERROR executing cell #{i+1}: {e}")
                    # Save partial outputs
                    with open(p, "w", encoding="utf-8") as f:
                        nbformat.write(nb, f)
                    raise

                elapsed = time.time() - t0
                safe_log(f"  Execution time: {elapsed:.2f}s | Outputs: {len(cell.outputs)}")

                # Inspect outputs
                for out in cell.outputs:
                    out_type = out.get("output_type")
                    if out_type == "stream":
                        text = out.get("text", "").strip()
                        # Print sample output
                        sample = text.splitlines()[0] if text else ""
                        if len(text.splitlines()) > 1:
                            sample += f" (+ {len(text.splitlines())-1} lines)"
                        safe_log(f"    [stdout] {sample}")
                    elif out_type in ["display_data", "execute_result"]:
                        data_keys = list(out.get("data", {}).keys())
                        safe_log(f"    [{out_type}] Rendered: {data_keys}")
                    elif out_type == "error":
                        ename = out.get("ename")
                        evalue = out.get("evalue")
                        safe_log(f"    [ERROR] {ename}: {evalue}")
                        with open(p, "w", encoding="utf-8") as f:
                            nbformat.write(nb, f)
                        raise RuntimeError(f"Cell #{i+1} failed with {ename}: {evalue}")

    total_time = time.time() - start_all
    safe_log(f"\nAll {len(code_cells)} code cells executed successfully in {total_time:.1f}s!")

    # Save executed notebook
    with open(p, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    safe_log(f"Persisted executed notebook with embedded outputs to: {p}")
    return nb

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "notebooks/01_dataset_exploration.ipynb"
    execute_notebook_with_kernel(target)
