from PyhtonImport import *


class CreateMask(nn.Module):
    def __init__(self, max_sequence_length, inf=-1e9):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.inf = inf

    def forward(self, source_batch, target_batch):
        num_sentences = len(source_batch)
        look_ahead_mask = torch.full([self.max_sequence_length, self.max_sequence_length] , True)
        look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
        encoder_padding_mask = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)
        decoder_padding_mask_self_attention = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)
        decoder_padding_mask_cross_attention = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)

        for idx in range(num_sentences):
            eng_sentence_length, kn_sentence_length = len(source_batch[idx]), len(target_batch[idx])
            eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, self.max_sequence_length)
            kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, self.max_sequence_length)
            encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
            encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
            decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
            decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
            decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
            decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

        encoder_self_attention_mask = torch.where(encoder_padding_mask, self.inf, 0)
        decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, self.inf, 0)
        decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, self.inf, 0)
        return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

# create_mask = CreateMask(max_sequence_length=200)
# encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_mask(("hello", "what is your name?"), ("hi", "my name is nimy"))
# print(encoder_self_attention_mask)