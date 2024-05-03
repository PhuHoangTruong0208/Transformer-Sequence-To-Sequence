from PyhtonImport import *

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden, dropout):
        super().__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=ffn_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=ffn_hidden, out_features=d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x