import pytest
import torch
import pandas as pd
from transformers import BertTokenizer
from src.data.dataset import NERDataset
from src.models.ewc import EWC
from transformers import BertForTokenClassification

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'sentence': ['This is a test'],
        'word_labels': ['O,O,O,O']
    })

@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def test_dataset_creation(sample_data, tokenizer):
    dataset = NERDataset(sample_data, tokenizer, max_len=10)
    assert len(dataset) == 1
    
    item = dataset[0]
    assert all(key in item for key in ['ids', 'mask', 'targets'])
    assert isinstance(item['ids'], torch.Tensor)
    assert isinstance(item['mask'], torch.Tensor)
    assert isinstance(item['targets'], torch.Tensor)

@pytest.fixture
def model():
    return BertForTokenClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=31
    )

@pytest.fixture
def ewc(model):
    return EWC(model, device='cpu')

def test_ewc_initialization(ewc):
    assert ewc.fisher_information == {}
    assert ewc.param_prior == {}

def test_compute_ewc_lambda(ewc):
    assert ewc.compute_ewc_lambda() == 0.0
    
    ewc.fisher_information = {
        'test': torch.tensor([1.0, 2.0, 3.0])
    }
    lambda_val = ewc.compute_ewc_lambda()
    assert lambda_val > 0