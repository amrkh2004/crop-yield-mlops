import glob
import json
import os


def check_notebook(path):
    print(f"Checking notebook: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for cell in nb.get("cells", []):
            source = "".join(cell.get("source", []))
            if "kaggle" in source.lower() or "yield_df" in source.lower() or "read_csv" in source.lower():
                print(f"--- Cell in {os.path.basename(path)} ---")
                print(source[:500])
    except Exception as e:
        print(f"Error reading {path}: {e}")


for nb_path in (
    glob.glob("E:/Downloads/*.ipynb") + glob.glob("E:/Downloads/*/*.ipynb") + glob.glob("E:/Downloads/*/*/*.ipynb")
):
    check_notebook(nb_path)
