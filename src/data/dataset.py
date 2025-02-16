import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label2id):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
    
    def __getitem__(self, index):
        sentence = self.data.sentence[index]
        word_labels = self.data.word_labels[index]
        
        tokenized_sentence, labels = self._tokenize_and_preserve_labels(
            sentence, word_labels
        )
        
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        labels = ["O"] + labels + ["O"]
        
        if len(tokenized_sentence) > self.max_len:
            tokenized_sentence = tokenized_sentence[:self.max_len]
            labels = labels[:self.max_len]
        else:
            padding_length = self.max_len - len(tokenized_sentence)
            tokenized_sentence.extend(['[PAD]'] * padding_length)
            labels.extend(['O'] * padding_length)
        
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokenized_sentence]
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_ids = [self.label2id[label] for label in labels]
        
        return {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(attention_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def __len__(self):
        return self.len
    
    def _tokenize_and_preserve_labels(self, sentence, word_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence.split(), word_labels.split(',')):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels