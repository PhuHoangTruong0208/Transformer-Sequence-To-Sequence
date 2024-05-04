from TransformersModel import TranformersModel
from CreateMask import CreateMask
from PyhtonImport import *


class GetIndex:
    def __init__(self, start_token='<start>', padding_token="<pad>", end_token="<end>"):
        self.START_TOKEN = start_token
        self.PADDING_TOKEN = padding_token
        self.END_TOKEN = end_token
        self.vocab = [self.START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@',
                        '[', '\\', ']', '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
                        "A", "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C",
                        "V", "B", "N", "M",
                        '{', '|', '}', '~', self.PADDING_TOKEN, self.END_TOKEN]

    def get_index(self):
        index_to_char = {k:v for k,v in enumerate(self.vocab)}
        char_to_index = {v:k for k,v in enumerate(self.vocab)}
        return index_to_char, char_to_index
    

class PreprocessingConversations:
    def __init__(self, limit_total_sentence=120, sequence_length=200):
        self.limit_total_sentence = limit_total_sentence
        self.sequence_length = sequence_length
        self.get_index = GetIndex()

    def read_file(self):
        with open(file="transformers/questions.txt", mode="r", encoding="utf-8") as file:
            source_sentences = file.read().splitlines()
        with open(file="transformers/answers.txt", mode="r", encoding="utf-8") as file:
            target_sentences = file.read().splitlines()
        source_sentences = source_sentences[:self.limit_total_sentence]
        target_sentences = target_sentences[:self.limit_total_sentence]
        return source_sentences, target_sentences
    
    def is_valid_token(self, sentence, vocab):
        for token in list(set(sentence)):
            if token not in vocab:
                return False
            else:
                return True
    
    def is_valid_length(self, sentence, sequence_length):
        return len(list(sentence)) < (sequence_length - 1)
    
    def process_data(self):
        source_sentences, target_sentences = self.read_file()
        valid_sentence_indicies = []
        for index in range(len(target_sentences)):
            target_sentence, source_sentence = target_sentences[index], source_sentences[index]
            if self.is_valid_length(target_sentence, self.sequence_length) \
                and self.is_valid_length(source_sentence, self.sequence_length) \
                and self.is_valid_token(target_sentence, self.get_index.vocab):
                    valid_sentence_indicies.append(index)
        source_sentences = [source_sentences[i] for i in valid_sentence_indicies]
        target_sentences = [target_sentences[i] for i in valid_sentence_indicies]
        return source_sentences, target_sentences
    

class TextDataset(Dataset):
    def __init__(self):
        self.source_sentence, self.target_sentence = PreprocessingConversations().process_data()

    def __len__(self):
        return len(self.source_sentence)
    
    def __getitem__(self, idx):
        return self.source_sentence[idx], self.target_sentence[idx]
    

class TransformersTraining:
    def __init__(self, d_model=512, batch_size=30, ffn_hidden=2048, num_heads=8, dropout=0.1, vocab_size=len(GetIndex().vocab),
                 num_layers=2, epochs=40):
        self.d_model = d_model
        self.batch_size = batch_size
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # thiết lập đối tượng
        self.get_index = GetIndex()
        self.preprocess = PreprocessingConversations()
        self.create_mask = CreateMask(max_sequence_length=self.preprocess.sequence_length)
        # setup model
        self.index_to_char, self.char_to_index = self.get_index.get_index()
        self.transformer_model = TranformersModel(d_model=d_model, ffn_hidden=ffn_hidden, num_heads=num_heads,
                                    dropout=dropout, num_layers=num_layers, to_index=self.char_to_index, max_sequence_length=self.preprocess.sequence_length, vocab_size=vocab_size,
                                    START_TOKEN=self.get_index.START_TOKEN, END_TOKEN=self.get_index.END_TOKEN, PADDING_TOKEN=self.get_index.PADDING_TOKEN)
        for params in self.transformer_model.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)
        # thiết lập hàm mất mát và hàm tối ưu
        self.criterian = nn.CrossEntropyLoss(ignore_index=self.char_to_index[self.get_index.PADDING_TOKEN],
                                reduction='none')
        self.optim = torch.optim.Adam(self.transformer_model.parameters(), lr=1e-4)

        self.epochs = epochs
        self.total_loss = 0
        self.transformer_model.train()
        self.transformer_model.to(device)

        dataset = TextDataset()
        self.train_loader = DataLoader(dataset, self.batch_size)

    def training(self):
        index_to_char, char_to_index = self.get_index.get_index()
        for epoch in range(self.epochs):
            print(f"Epoch : {epoch}")
            iterator = iter(self.train_loader)
            for batch_num, batch in enumerate(iterator):
                self.transformer_model.train()
                source_batch, target_batch = batch
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_mask(source_batch, target_batch)
                self.optim.zero_grad()
                target_predict = self.transformer_model(source_batch, target_batch, encoder_self_attention_mask.to(device),
                                                        decoder_self_attention_mask.to(device), decoder_cross_attention_mask.to(device),
                                                        enc_start_token=False, enc_end_token=False, dec_start_token=True, dec_end_token=True)
                labels = self.transformer_model.decoder_layer.embedding.batch_tokenize(batch=target_batch, start_token=False, end_token=True)
                loss = self.criterian(
                    target_predict.view(-1, self.vocab_size).to(device),
                    labels.view(-1).to(device)
                ).to(device)
                valid_indicies = torch.where(labels.view(-1) == char_to_index[self.get_index.PADDING_TOKEN], False, True)
                loss = loss.sum() / valid_indicies.sum()
                loss.backward()
                self.optim.step()
                
                if batch_num % 100 == 0:
                    print(f"Iteration {batch_num} : {loss.item()}")
                    print("X: hello what is your name?")
                    print("Predict: ", end="", flush=True)

                    source_senttence = ("hello what is your name?",)
                    target_sentence = ("", )
                    for word_counter in range(int(self.preprocess.sequence_length/1.5)):
                        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_mask(source_senttence, target_sentence)
                        target_predict_test = self.transformer_model(source_senttence, target_sentence, encoder_self_attention_mask.to(device),
                                                        decoder_self_attention_mask.to(device), decoder_cross_attention_mask.to(device),
                                                        enc_start_token=False, enc_end_token=False, dec_start_token=True, dec_end_token=True)
                        next_token_prob_distribution = target_predict_test[0][word_counter]
                        next_token_index = torch.argmax(next_token_prob_distribution).item()
                        next_token = self.index_to_char[next_token_index]
                        target_sentence = (target_sentence[0] + next_token, )
                        if next_token == self.get_index.END_TOKEN:
                            break
                        print(next_token, end="", flush=True)
                    print()

    def predict(self, source_sequence):
        index_to_char, char_to_index = self.get_index.get_index()
        source_sequence = (source_sequence, )
        target_sequence = ("", )
        try:
            for word_counter in range(self.preprocess.sequence_length-1):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_mask(source_sequence, target_sequence)
                # dự đoán của mô hình
                target_predict = self.transformer_model(source_sequence, target_sequence, encoder_self_attention_mask.to(device),
                                                        decoder_self_attention_mask.to(device), decoder_cross_attention_mask.to(device),
                                        enc_start_token=False, enc_end_token=False, dec_start_token=True, dec_end_token=True)
                next_token_prob_distribution = target_predict[0][word_counter]
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_char[next_token_index]
                target_sequence = (target_sequence[0] + next_token, )
                if next_token == self.get_index.END_TOKEN:
                    break
                print(next_token, end="", flush=True)
        except:
            pass
        print()

transformers = TransformersTraining(num_heads=8, num_layers=1, d_model=256, ffn_hidden=512, dropout=0.01, epochs=100)
transformers.training()
while True:
    user_input = input("bạn : ")
    predict = transformers.predict(source_sequence=user_input)
