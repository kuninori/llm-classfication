from pathlib import Path
import json
def read_taglist():
    path = Path("./data/livedoor-tags.json")
    with open(path, "r") as f:
        return json.load(f)