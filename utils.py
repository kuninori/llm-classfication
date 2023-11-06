import json
from pathlib import Path

def load_json(str):
    Path(str)
    with file.open() as f:
        data = json.load(f)
    return data