import logging
import argparse
from pathlib import Path
import wandb
from src.config import ModelConfig, TrainingConfig, ENTITY_SET
from src.models.bert_model import NERModel
from src.models.ewc import EWC
from src.training.trainer import NERTrainer
from src.training.utils import prepare_datasets, load_and_preprocess_data, save_metrics
from src.preprocessing.preprocessor import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train NER model with EWC')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing the dataset files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save model checkpoints and metrics')
    parser.add_argument('--wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.wandb:
        wandb.init(project=TrainingConfig.wandb_project_name)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    logger.info("Initializing model...")
    model = NERModel(model_config)
    
    preprocessor = DataPreprocessor(ENTITY_SET)
    
    data_dir = Path(args.data_dir)
    datasets = []
    for i in range(1, training_config.num_datasets + 1):
        file_path = data_dir / f"data{i}.xlsx"
        df = load_and_preprocess_data(file_path, training_config.drop_columns)
        df = preprocessor.preprocess_dataset(df, model.tokenizer, training_config.max_len)
        datasets.append(df)
        logger.info(f"Loaded dataset data{i}: {len(df)} samples")
    
    ewc = EWC(model.model, model_config.device)
    keep_loader = None
    
    for task_id, df in enumerate(datasets, 1):
        logger.info(f"Training on task {task_id}")
        
        train_loader, test_loader, keep_loader = prepare_datasets(
            df, model.tokenizer, training_config, preprocessor, keep_loader
        )
        
        trainer = NERTrainer(
            model.model,
            training_config,
            model_config.device,
            ewc if task_id > 1 else None
        )
        
        metrics = trainer.train(train_loader, test_loader, keep_loader)
        
        metrics_path = output_dir / f"metrics_task_{task_id}.csv"
        save_metrics(metrics, metrics_path)
        
        model_path = output_dir / f"model_task_{task_id}"
        model.save_model(model_path)
        
        if task_id < len(datasets):
            logger.info("Computing Fisher Information for next task...")
            ewc.compute_fisher_information(train_loader)
            ewc.store_model_parameters()
        
        if args.wandb:
            wandb.log({
                f"task_{task_id}/final_train_loss": metrics['train_loss'][-1],
                f"task_{task_id}/final_eval_f1": metrics['eval_f1'][-1]
            })
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()