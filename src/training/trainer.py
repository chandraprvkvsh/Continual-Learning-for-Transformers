import torch
from typing import Tuple, Dict, Optional, List
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report
from tqdm import tqdm
import numpy as np
from src.config import TrainingConfig
from src.models.ewc import EWC
from src.preprocessing.preprocessor import DataPreprocessor

class NERTrainer:
    def __init__(
        self,
        model,
        config: TrainingConfig,
        device: str,
        ewc: Optional[EWC] = None,
        preprocessor: DataPreprocessor = None
    ):
        self.model = model
        self.config = config
        self.device = device
        self.ewc = ewc
        self.id2label = preprocessor.id2label if preprocessor else {}
    
    def train_epoch(
        self,
        training_loader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        tr_loss, tr_accuracy = 0, 0
        nb_tr_steps = 0
        
        self.model.train()
        for batch in tqdm(training_loader, desc="Training", leave=False):
            ids = batch['ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            outputs = self.model(input_ids=ids, attention_mask=mask, labels=targets)
            loss = outputs.loss
            
            if self.ewc is not None:
                ewc_loss = self.ewc.compute_ewc_loss()
                loss += self.ewc.compute_ewc_lambda() * ewc_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            tr_loss += loss.item()
            nb_tr_steps += 1
            
            active_logits = outputs.logits.view(-1, self.model.num_labels)
            flattened_targets = targets.view(-1)
            active_accuracy = mask.view(-1) == 1
            
            predictions = torch.argmax(active_logits, axis=1)
            masked_targets = torch.masked_select(flattened_targets, active_accuracy)
            masked_predictions = torch.masked_select(predictions, active_accuracy)
            
            tr_accuracy += accuracy_score(
                masked_targets.cpu().numpy(),
                masked_predictions.cpu().numpy()
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            optimizer.step()
        
        return tr_loss / nb_tr_steps, tr_accuracy / nb_tr_steps
    
    def evaluate(
        self,
        eval_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        eval_predictions = []
        eval_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                ids = batch['ids'].to(self.device)
                mask = batch['mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(input_ids=ids, attention_mask=mask, labels=targets)
                loss = outputs.loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                eval_loss += loss.item()
                nb_eval_steps += 1
                
                active_logits = outputs.logits.view(-1, self.model.num_labels)
                flattened_targets = targets.view(-1)
                active_accuracy = mask.view(-1) == 1
                
                predictions = torch.argmax(active_logits, axis=1)
                
                masked_targets = torch.masked_select(flattened_targets, active_accuracy)
                masked_predictions = torch.masked_select(predictions, active_accuracy)
                
                eval_labels.extend(masked_targets)
                eval_predictions.extend(masked_predictions)
        
        label_names = [self.id2label.get(id, 'O') for id in eval_labels]
        pred_names = [self.id2label.get(id, 'O') for id in eval_predictions]
        
        eval_loss = eval_loss / nb_eval_steps
        metrics = classification_report([label_names], [pred_names], output_dict=True, zero_division=0)
        
        return eval_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        keep_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'eval_loss': [],
            'eval_f1': []
        }
        
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            eval_loss, metrics = self.evaluate(eval_loader)
            history['eval_loss'].append(eval_loss)
            history['eval_f1'].append(metrics['macro avg']['f1-score'])
            
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
            print(f"Eval Loss: {eval_loss:.4f} | Eval F1: {metrics['macro avg']['f1-score']:.4f}")
        
        if keep_loader is not None:
            print("Training on preserved samples...")
            self.train_epoch(keep_loader, optimizer)
        
        return history