import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CaptionGenerationNet(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super(CaptionGenerationNet, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.fc_1 = nn.Linear(2048, 256) 
        self.embedding = nn.Embedding(vocab_size, 200) 
        self.lstm = nn.LSTM(200, 256, batch_first=True) 
        self.fc_2 = nn.Linear(256, 256) 
        self.fc_3 = nn.Linear(256, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        x, y = input

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.drop(x)

        y = self.embedding(y)
        y = self.drop(y)

        _, (h_n, _) = self.lstm(y)
        h_n = h_n.squeeze(0)

        z = x + h_n

        z = self.fc_2(z)
        z = self.relu(z)
        z = self.fc_3(z)

        return z








