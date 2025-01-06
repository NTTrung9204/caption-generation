from torch.utils.data import Dataset
import numpy as np

class SeqDataset(Dataset):
    def __init__(self, X1: list[np.ndarray[np.float64]], X2: list[np.ndarray[np.int64]], Y: list[np.ndarray[np.int64]]) -> None:
        self.X1 = X1
        self.X2 = X2
        self.Y = Y

    def __len__(self) -> int:
        return len(self.X1)

    def __getitem__(self, index) -> tuple[tuple[np.ndarray[np.float64], np.ndarray[np.int64]], np.ndarray[np.int64]]:
        return (self.X1[index], self.X2[index]), self.Y[index]