from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, processed_caption: dict[str, list[str]]) -> None:
        self.processed_caption = processed_caption

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass