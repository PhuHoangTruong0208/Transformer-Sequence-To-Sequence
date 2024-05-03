from PyhtonImport import *
from TranformersEncoder import Encoder
from TranformersDecoder import Decoder

class TranformersModel(nn.Module):

    def __init__(self, d_model, num_heads, ffn_hidden, dropout, to_index, max_sequence_length, num_layers, START_TOKEN,
                END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.encoder_layer = Encoder(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout,
                            num_layers=num_layers, to_index=to_index, max_sequence_length=max_sequence_length, START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, PADDING_TOKEN=PADDING_TOKEN)
        self.decoder_layer = Decoder(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout,
                            num_layers=num_layers, to_index=to_index, max_sequence_length=max_sequence_length, START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, PADDING_TOKEN=PADDING_TOKEN)
        self.out_linear = nn.Linear(in_features=d_model, out_features=len(to_index)+1)

    def forward(self, x, y, encoder_mask, decoder_mask, decoder_cross_mask, enc_start_token, enc_end_token,
                dec_start_token, dec_end_token):
        x = self.encoder_layer(x, encoder_mask=encoder_mask, start_token=enc_start_token, end_token=enc_end_token)
        y = self.decoder_layer(x, y, decoder_mask=decoder_mask, decoder_cross_mask=decoder_cross_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.out_linear(y)
        return out
    
# x = ("hello", "what is your name?")
# y = ("hi", "my name is nimy")
# index_to = dict(enumerate(["h", "e", "l", "o", "w", "a", "t", "y", "o", "u", "r", "n", "m", "e", "?", " ", "i", "s"]+["<s>", "<e>", "<p>"]))
# to_index = {v:k for k, v in index_to.items()}

# dec = TranformersModel(d_model=512, num_heads=8, ffn_hidden=1024, dropout=0.1, num_layers=2, to_index=to_index,
#               max_sequence_length=100, START_TOKEN="<s>", END_TOKEN="<e>", PADDING_TOKEN="<p>")
# result = dec(x, y, encoder_mask=None, decoder_mask=None, decoder_cross_mask=None, enc_start_token=False, enc_end_token=True,
#              dec_start_token=False, dec_end_token=False)
# print(result)