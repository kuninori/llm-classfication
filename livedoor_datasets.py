from torch.utils.data import Dataset
from pathlib import Path

class LivedoorDataset(Dataset):
    def __init__(self):
        l = list(Path("./data/livedoor/text").glob("*/*.txt"))
        self.list = l
    def __getitem__(self, index):
        path = self.list[index]
        with path.open("r") as f:
            text = f.read()
            text = text.replace("\n\n\n", "\n")
            lines = text.split("\n")
            text = "\n".join(lines[2:])
            f.close()

            paths = path.parts
            label = paths[-2]

        return (text, label)
    def __len__(self):
        return len(self.list)

d = LivedoorDataset()