from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
root = Path.cwd()
path = root / 'Logistic_Regression/python.ipynb'
nb = nbformat.read(path, as_version=4)
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(nb, km=km, timeout=180, resources={'metadata': {'path': str(path.parent)}})
client.execute()
nbformat.validate(nb)
nbformat.write(nb, path)
print('EXECUTED', sum(c.cell_type == 'code' for c in nb.cells), 'code cells')
print('ERRORS', sum(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs))
