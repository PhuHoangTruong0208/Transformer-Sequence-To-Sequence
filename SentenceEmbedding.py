from PyhtonImport import *


# mã hóa vị trí giúp các vector tokens chứa các mã hóa vị trí giúp mô hình nhớ được vị trí của từng từ
def position_encoding(d_model, max_sequence_length):
    even_i = torch.arange(start=0, end=d_model, step=2, dtype=torch.float32)
    denominator = torch.pow(10000, even_i/d_model)
    position = torch.arange(max_sequence_length).reshape(max_sequence_length, 1)
    even_PE = torch.sin(position / denominator)
    odd_PE = torch.cos(position / denominator)
    stacked = torch.stack([even_PE, odd_PE], dim=2)
    return stacked.reshape(max_sequence_length, d_model).to(device)


# chuyển lô thành các tokens để chuẩn bị đưa vào chạy qua mô hình
class SentenceEmbedding(nn.Module):

    def __init__(self, d_model, max_sequence_lenght, to_index, dropout, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_lenght = max_sequence_lenght
        self.char_to_index = to_index
        self.dropout = dropout
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.embedding = nn.Embedding(num_embeddings=len(to_index)+1, embedding_dim=d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            tokens_sentence = [self.char_to_index[token] for token in list(sentence)]
            if start_token:
                tokens_sentence.insert(0, self.char_to_index[self.START_TOKEN])
            if end_token:
                tokens_sentence.append(self.char_to_index[self.END_TOKEN])
            for _ in range(len(tokens_sentence), self.max_sequence_lenght):
                tokens_sentence.append(self.char_to_index[self.PADDING_TOKEN])
            return torch.tensor(tokens_sentence).to(device)
        
        tokenized = []
        for sentence in batch:
            tokenized.append(tokenize(sentence, start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(device)
    
    def forward(self, x, start_token, end_token):
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        x = self.dropout(x + position_encoding(self.d_model, self.max_sequence_lenght))
        return x.to(device)

# x = ("hello", "what is your name?")
# index_to = dict(enumerate(["h", "e", "l", "o", "w", "a", "t", "y", "o", "u", "r", "n", "m", "e", "?", " ", "i", "s"]+["<s>", "<e>", "<p>"]))
# to_index = {v:k for k, v in index_to.items()}
# embed = SentenceEmbedding(d_model=512, max_sequence_lenght=200, char_to_index=to_index, dropout=0.1,
#                           START_TOKEN="<s>", END_TOKEN="<e>", PADDING_TOKEN="<p>")
# print(embed(x, start_token=True, end_token=True))