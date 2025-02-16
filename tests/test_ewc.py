import pytest
import torch
from transformers import BertForTokenClassification
from src.models.ewc import EWC

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
