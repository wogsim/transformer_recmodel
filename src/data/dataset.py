from torch.utils.data import Dataset
from src.modeling.utils import convert_dict_to_tensor
import polars as pl

class LavkaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> 'LavkaDataset':
        converted_data = [convert_dict_to_tensor(group) for group in df.to_struct()]
        return cls(converted_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]