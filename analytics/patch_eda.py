import json

filepath = '/workspace/EDA_advanced_eligibilite_v6.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

patched = False
for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if '"ratio_volatilite_solde": ("solde_volatilite_indefinie", "solde_moyen")' in line:
                source[i] = line.replace('"solde_volatilite_indefinie"', '"solde_volatilite_relative_imp"')
                patched = True
        cell['source'] = source

if patched:
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook patched successfully.")
else:
    print("Could not find the target line to patch.")
