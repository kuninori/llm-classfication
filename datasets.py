from torch.utils.data import Dataset
import os

import utils


class TextDataset(Dataset):
    def __init__(self):
        self.list = os.listdir("./data")
    def __getitem__(self, index):
        data = utils.load_json(self.list[index])
        print(data)
        return data
    def __len__(self):
        return len(self.list)