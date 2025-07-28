import nbformat
import os

notebook_filename = 'regressao.ipynb'

if not os.path.exists(notebook_filename):
    print(f"Notebook file '{notebook_filename}' not found in the current directory.")
    exit(1)

print(f"Loading notebook: {notebook_filename}")
with open(notebook_filename, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Find section boundaries
section_indices = []
section_names = [
    '# Regressão Linear - Modelo Inicial',
    '# Modelo Regional',
    '# Modelo por Estado - Com Erro', 
    '# Modelo por Estado - Corrigido'
]

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown':
        for section_name in section_names:
            if cell.source.strip().startswith(section_name):
                section_indices.append((i, section_name))
                print(f"Found '{section_name}' at cell index: {i}")
                break

if len(section_indices) != 4:
    print(f"Expected 4 sections, but found {len(section_indices)}:")
    for idx, name in section_indices:
        print(f"  - {name} at index {idx}")
    exit(1)

# Define section boundaries
sections = []
for i in range(len(section_indices)):
    start_idx = section_indices[i][0]
    if i < len(section_indices) - 1:
        end_idx = section_indices[i + 1][0]
    else:
        end_idx = len(nb.cells)
    
    section_name = section_indices[i][1].replace('# ', '').replace(' - ', '-').replace(' ', '_').lower()
    sections.append((start_idx, end_idx, section_name))

print(f"\nCreating {len(sections)} notebook sections:")

# Create individual notebooks for each section
for i, (start_idx, end_idx, section_name) in enumerate(sections):
    section_cells = nb.cells[start_idx:end_idx]
    
    # Create new notebook with section cells
    nb_section = nbformat.v4.new_notebook()
    nb_section.cells = section_cells
    nb_section.metadata = nb.metadata
    
    # Generate filename
    section_filename = f'regressao-parte{i+1}-{section_name}.ipynb'
    
    # Write the notebook
    with open(section_filename, 'w', encoding='utf-8', errors='surrogatepass') as f:
        nbformat.write(nb_section, f)
    
    print(f"Created: {section_filename} (cells {start_idx}-{end_idx-1}, {len(section_cells)} cells)")

print(f"\n✅ Successfully split {notebook_filename} into {len(sections)} sections!")
