from transformers import BertTokenizer, BertForTokenClassification
from src.config import ModelConfig

class NERModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = BertForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        self.model.to(config.device)
    
    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: str):
        self.model = BertForTokenClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to(self.config.device)