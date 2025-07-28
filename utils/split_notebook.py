import nbformat
import os

notebook_filename = 'analise-municipios.ipynb'

if not os.path.exists(notebook_filename):
    print(f"Notebook file '{notebook_filename}' not found in the current directory.")
    exit(1)

print(f"Loading notebook: {notebook_filename}")
with open(notebook_filename, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

start_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and cell.source.strip().startswith('# Parte 5'):
        start_idx = i
        break

if start_idx is None:
    print("Could not find a markdown cell starting with '# Parte 5'.")
    exit(1)

print(f"'Parte 5' found at cell index: {start_idx}")

parte5_cells = nb.cells[start_idx:]
remaining_cells = nb.cells[:start_idx]

# Copy metadata from the original notebook
nb_parte5 = nbformat.v4.new_notebook()
nb_parte5.cells = parte5_cells
nb_parte5.metadata = nb.metadata

nb_remaining = nbformat.v4.new_notebook()
nb_remaining.cells = remaining_cells
nb_remaining.metadata = nb.metadata

parte5_filename = 'analise-municipios-parte5.ipynb'
remaining_filename = 'analise-municipios-parte1-4.ipynb'

with open(parte5_filename, 'w', encoding='utf-8', errors='surrogatepass') as f:
    nbformat.write(nb_parte5, f)
print(f"Created: {parte5_filename}")

with open(remaining_filename, 'w', encoding='utf-8', errors='surrogatepass') as f:
    nbformat.write(nb_remaining, f)
print(f"Created: {remaining_filename}")

