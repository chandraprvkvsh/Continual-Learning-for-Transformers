from typing import List, Dict
import pandas as pd
from src.config import ENTITY_SET, label2id, id2label

class DataPreprocessor:
    def __init__(self, entityset: List[str] = ENTITY_SET):
        self.entityset = entityset
        self.label2id = label2id
        self.id2label = id2label
    
    def extract_words(self, row: pd.Series) -> List[Dict[str, str]]: 
        if not isinstance(row['text'], str):
            return []
        
        extracted_words = []
        tags = row['tags'].split(',')
        
        for tag in tags:
            tag_elements = tag.split(':')
            if len(tag_elements) != 3:
                continue
            
            start, end, label = tag_elements
            start, end = int(start), int(end)
            word = row['text'][start-1:end]
            
            if word and not word[-1].isalpha():
                word = word[:-1]
            
            extracted_words.append({word: label})
        
        return extracted_words

    def preprocess_dataset(self, dataframe: pd.DataFrame, tokenizer, max_len: int) -> pd.DataFrame:
        dataframe['text'] = dataframe['text'].astype(str)
        dataframe['labels'] = dataframe.apply(lambda row: self.extract_words(row), axis=1)
        dataframe['text_ids'] = dataframe['text'].apply(lambda text: tokenizer.tokenize(text))
        dataframe['label_ids'] = dataframe.apply(lambda row: self.generate_label_ids(row['labels'], row['text_ids']), axis=1)
        dataframe['sentence'] = dataframe['text_ids'].apply(lambda tokens: ' '.join(tokens))
        dataframe['word_labels'] = dataframe['label_ids'].apply(lambda labels: ','.join(labels))
        dataframe.drop(["text", "labels", "text_ids", "label_ids", "tags"], axis=1, inplace=True)
        return dataframe

    def generate_label_ids(self, labels: List[Dict[str, str]], text_ids: List[str]) -> List[str]:
        label_ids = []
        for token in text_ids:
            label_ids.append(self.label2id.get(token, 'O'))
        return label_ids