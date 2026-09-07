import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                       padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
        last_hidden = self.dropout(hidden[-1])
        return self.fc(last_hidden).squeeze(1)
