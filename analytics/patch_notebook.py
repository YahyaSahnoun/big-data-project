import json

filepath = '/workspace/pipeline_v1_11_V2.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if 'COL_HAUTE_CARDINALITE = _cfg["col_haute_cardinalite"]' in line and 'COLS_CATEGORIELLES_BASSE_CARDINALITE =' in "".join(source):
                # Inject the categorical array appends before this line
                inject = (
                    "    for new_c in [\"pack_actuel_x_CUSTOMER_RATING\", \"pack_etat_x_CUSTOMER_RATING\", \"pack_actuel_x_pack_etat\"]:\n"
                    "        if new_c not in COLS_CATEGORIELLES_BASSE_CARDINALITE:\n"
                    "            COLS_CATEGORIELLES_BASSE_CARDINALITE.append(new_c)\n"
                )
                if inject not in "".join(source):
                    source.insert(i, inject)
            
            if 'VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")' in line:
                source[i] = line.replace('handleInvalid="skip"', 'handleInvalid="keep"')
                print("Patched handleInvalid=skip to keep.")
                
        cell['source'] = source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook patched successfully.")
