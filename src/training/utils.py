from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import NERDataset
from src.preprocessing.preprocessor import DataPreprocessor

def prepare_datasets(
    df: pd.DataFrame,
    tokenizer,
    config,
    preprocessor: DataPreprocessor,
    keep_loader=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Handle NaN and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    train_df = df.sample(frac=config.train_size, random_state=config.random_state)
    test_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    
    keep_df = train_df.sample(n=config.keep_sample_size, random_state=config.random_state).reset_index(drop=True)
    
    train_dataset = NERDataset(train_df, tokenizer, config.max_len, preprocessor.label2id)
    test_dataset = NERDataset(test_df, tokenizer, config.max_len, preprocessor.label2id)
    keep_dataset = NERDataset(keep_df, tokenizer, config.max_len, preprocessor.label2id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False
    )
    
    if keep_loader is None:
        keep_loader = DataLoader(
            keep_dataset,
            batch_size=config.keep_batch_size,
            shuffle=True
        )
    
    return train_loader, test_loader, keep_loader

def load_and_preprocess_data(file_path: str, drop_columns: List[str]) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.drop(drop_columns, axis=1, inplace=True)
    # Handle NaN and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def save_metrics(metrics: Dict[str, List[float]], file_path: str):
    pd.DataFrame(metrics).to_csv(file_path, index=False)