from PyhtonImport import *
from MultiheadAttention import MultiheadAttention, MultiheadCrossAttention
from SentenceEmbedding import SentenceEmbedding
from PositionwiseFeedForward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, ffn_hidden, dropout):
        super().__init__()
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = LayerNorm(normalized_shape=d_model)
        self.cross_attention = MultiheadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layernorm2 = LayerNorm(normalized_shape=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_hidden=ffn_hidden, dropout=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.layernorm3 = LayerNorm(normalized_shape=d_model)
    
    def forward(self, x, y, decoder_mask, decoder_cross_mask):
        pre_y = y.clone()
        y = self.attention(y, mask=decoder_mask)
        y = self.dropout1(y)
        y = self.layernorm1(y + pre_y)
        pre_y = y.clone()
        y = self.cross_attention(x, y, mask=decoder_cross_mask)
        y = self.dropout2(y)
        y = self.layernorm2(y + pre_y)
        pre_y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layernorm3(y + pre_y)
        return y


class SequentialDecoder(nn.Module):

    def __init__(self, d_model, num_heads, ffn_hidden, dropout, num_layers):
        super().__init__()
        self.layers = [DecoderLayer(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout)
                       for _ in range(num_layers)]
        
    def forward(self, x, y, decoder_mask, decoder_cross_mask):
        for layer in self.layers:
            y = layer(x, y, decoder_mask=decoder_mask, decoder_cross_mask=decoder_cross_mask)
        return y
    

class Decoder(nn.Module):
    
    def __init__(self, d_model, num_heads, ffn_hidden, dropout, num_layers, to_index, max_sequence_length,
                START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.embedding = SentenceEmbedding(d_model=d_model, max_sequence_lenght=max_sequence_length, to_index=to_index,
                                    dropout=dropout, START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, PADDING_TOKEN=PADDING_TOKEN)
        self.decoder_layer = SequentialDecoder(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden,
                                        dropout=dropout, num_layers=num_layers)
    
    def forward(self, x, y, decoder_mask, decoder_cross_mask, start_token, end_token):
        y = self.embedding(y, start_token=start_token, end_token=end_token)
        y = self.decoder_layer(x, y, decoder_mask=decoder_mask, decoder_cross_mask=decoder_cross_mask)
        return y
    

# y = ("hello", "what is your name?")
# x = torch.randn(2, 100, 512)
# index_to = dict(enumerate(["h", "e", "l", "o", "w", "a", "t", "y", "o", "u", "r", "n", "m", "e", "?", " ", "i", "s"]+["<s>", "<e>", "<p>"]))
# to_index = {v:k for k, v in index_to.items()}

# dec = Decoder(d_model=512, num_heads=8, ffn_hidden=1024, dropout=0.1, num_layers=2, to_index=to_index,
#               max_sequence_length=100, START_TOKEN="<s>", END_TOKEN="<e>", PADDING_TOKEN="<p>")
# result = dec(x, y, decoder_mask=None, decoder_cross_mask=None, start_token=False, end_token=True)
# print(result)