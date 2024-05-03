from PyhtonImport import *
from MultiheadAttention import MultiheadAttention
from SentenceEmbedding import SentenceEmbedding
from PositionwiseFeedForward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout):
        super().__init__()
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = LayerNorm(normalized_shape=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_hidden=ffn_hidden, dropout=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layernorm2 = LayerNorm(normalized_shape=d_model)
    
    def forward(self, x, encoder_mask):
        pre_x = x.clone()
        x = self.attention(x, mask=encoder_mask)
        x = self.dropout1(x)
        x = self.layernorm1(x + pre_x)
        pre_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.layernorm2(x + pre_x)
        return x


class SequentialEnocder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout, num_layers):
        super().__init__()
        self.layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout)
                  for _ in range(num_layers)]
    
    def forward(self, x, encoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_mask=encoder_mask)
        return x
    

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout, num_layers, to_index, max_sequence_length, START_TOKEN, 
                END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.embedding = SentenceEmbedding(d_model=d_model, max_sequence_lenght=max_sequence_length, to_index=to_index,
                                    dropout=dropout, START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, PADDING_TOKEN=PADDING_TOKEN)
        self.encoder_layer = SequentialEnocder(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout, num_layers=num_layers)

    def forward(self, x, encoder_mask, start_token, end_token):
        x = self.embedding(x, start_token=start_token, end_token=end_token)
        x = self.encoder_layer(x, encoder_mask=encoder_mask)
        return x


# x = ("hello", "what is your name?")
# index_to = dict(enumerate(["h", "e", "l", "o", "w", "a", "t", "y", "o", "u", "r", "n", "m", "e", "?", " ", "i", "s"]+["<s>", "<e>", "<p>"]))
# to_index = {v:k for k, v in index_to.items()}

# enc = Encoder(d_model=512, num_heads=8, ffn_hidden=1024, dropout=0.1, num_layers=2, to_index=to_index,
#               max_sequence_length=100, START_TOKEN="<s>", END_TOKEN="<e>", PADDING_TOKEN="<p>")
# result = enc(x, encoder_mask=None, start_token=False, end_token=True)
# print(result)