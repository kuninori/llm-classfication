from torch.utils.data import Dataset
import os
from pathlib import Path
import csv
import utils


class TextDataset(Dataset):
    def __init__(self):
        list = []
        for text_path in Path("./data/texts").iterdir():
            tag_path = Path("./data/tags") / f"{text_path.stem}.csv"
            if tag_path.is_file():
                list.append((text_path, tag_path))

        self.list = list

    def __getitem__(self, index):
        (text_path, tag_path) = self.list[index]
        with text_path.open("r") as f:
            text = f.read()
            f.close()
        with tag_path.open("r") as f:
            for row in csv.reader(f):
                t = row[0]
        return (text, t)

    def __len__(self):
        return len(self.list)
