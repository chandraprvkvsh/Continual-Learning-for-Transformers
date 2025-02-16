from dataclasses import dataclass, field
import torch
from src.preprocessing.utils import create_label_dicts

@dataclass
class TrainingConfig:
    max_len: int = 128
    train_batch_size: int = 32
    valid_batch_size: int = 16
    keep_batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-05
    max_grad_norm: float = 10
    keep_sample_size: int = 10
    train_size: float = 0.8
    num_datasets: int = 3
    random_state: int = 42
    drop_columns: list = field(default_factory=lambda: ["Unnamed: 0", "ID"])
    wandb_project_name: str = "ewc-ner"

@dataclass
class ModelConfig:
    model_name: str = 'bert-base-uncased'
    num_labels: int = 31
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

ENTITY_SET = [
    "treatment",
    "chronic_disease",
    "cancer",
    "gender",
    "pregnancy",
    "allergy_name",
    "contraception_consent",
    "language_literacy",
    "technology_access",
    "ethnicity",
    "attribute_clinical_variable",
    "age",
    "body_mass_index",
    "limit_upper_bound",
    "lower_bound"
]

label2id, id2label = create_label_dicts(ENTITY_SET)