import torch
from typing import Dict
from torch.utils.data import DataLoader

class EWC:
    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.param_prior: Dict[str, torch.Tensor] = {}
    
    def compute_fisher_information(self, dataloader: DataLoader) -> None:
        """Compute Fisher Information matrix for model parameters."""
        self.fisher_information = {}
        self.model.train()
        
        for batch in dataloader:
            ids = batch['ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(input_ids=ids, attention_mask=mask, labels=targets)
            loss = outputs.loss
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name not in self.fisher_information:
                        self.fisher_information[name] = (param.grad.detach() ** 2).clone()
                    else:
                        self.fisher_information[name] += (param.grad.detach() ** 2).clone()
    
    def compute_ewc_lambda(self, scale_factor: float = 1.0) -> float:
        """Compute EWC lambda based on Fisher Information magnitude."""
        if not self.fisher_information:
            return 0.0
        
        total_fisher = sum(torch.sum(value) for value in self.fisher_information.values())
        return scale_factor / total_fisher if total_fisher > 0 else 0.0
    
    def store_model_parameters(self) -> None:
        """Store current model parameters."""
        self.param_prior = {}
        for name, param in self.model.named_parameters():
            self.param_prior[name] = param.detach().clone()
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC loss term."""
        loss = torch.tensor(0., device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.param_prior:
                loss += (self.fisher_information[name] * 
                        (param - self.param_prior[name].to(self.device)) ** 2).sum()
        return loss