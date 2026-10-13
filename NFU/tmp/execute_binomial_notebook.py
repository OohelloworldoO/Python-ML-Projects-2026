from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
p=Path('Binomial_Normal/python.ipynb').resolve()
nb=nbformat.read(p,as_version=4)
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=120,resources={'metadata':{'path':str(p.parent)}}).execute()
nbformat.validate(nb)
assert all(o.output_type!='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
nbformat.write(nb,p)
for c in nb.cells:
    if c.cell_type=='code':
        for o in c.outputs:
            if o.output_type=='stream':
                print(o.text)
print('Notebook executed and saved successfully.')
