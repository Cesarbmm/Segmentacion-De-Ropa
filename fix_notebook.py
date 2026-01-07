import json
import os

notebook_path = r"c:/Users/pc/Desktop/Proyecto_Segmentacion/modelo_supremo.ipynb"

def fix_notebook():
    if not os.path.exists(notebook_path):
        print(f"Error: File not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    changed = False
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if 'path_save = "dataset\\label_list_clean_with_numeric.csv"' in line:
                    print("Found target line.")
                    line = line.replace('dataset\\label_list_clean_with_numeric.csv', 'dataset/label_list_clean_with_numeric.csv')
                    changed = True
                new_source.append(line)
            cell['source'] = new_source

    if changed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("Notebook fixed successfully.")
    else:
        print("Target line not found.")

if __name__ == "__main__":
    fix_notebook()
