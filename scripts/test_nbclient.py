import nbformat
from nbclient import NotebookClient

nb = nbformat.v4.new_notebook()
nb.cells.append(nbformat.v4.new_code_cell("print('Hello from real Jupyter kernel!')\nx = 42"))
nb.cells.append(nbformat.v4.new_code_cell("print('Result x:', x)"))

client = NotebookClient(nb, timeout=60, kernel_name='python3')
client.execute()

for idx, cell in enumerate(nb.cells):
    print(f"Cell {idx} output: {cell.outputs[0]['text'].strip()}")
