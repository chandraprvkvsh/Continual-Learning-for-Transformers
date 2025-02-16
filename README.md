# Continual Learning for Named Entity Recognition

## Overview

This project explores Named Entity Recognition (NER) using continual learning techniques. The model is trained on multiple datasets (data1, data2, etc.), where each dataset corresponds to a different NER task. The number of tasks (datasets) is configurable, making the approach highly flexible. Elastic Weight Consolidation (EWC) is used to address catastrophic forgetting.

## Approach

- **Continual Learning**: The model sequentially learns multiple NER tasks on different datasets.
- **Configurable Training**: Users can configure the number of datasets, entity set, and preprocessing mechanisms.
- **EWC Mechanism**: Fisher Information is computed to retain knowledge from previous tasks.

## Importance of Continual Learning

Continual Learning is crucial because it allows the same model to retain old knowledge while learning new knowledge. This means that once the model is trained on previous tasks, it can still remember and perform well on those tasks while also learning to handle new tasks. This is particularly important in dynamic environments where new data and tasks are continuously introduced.

## Elastic Weight Consolidation (EWC)

Elastic Weight Consolidation (EWC) is a method used for continual learning, where a neural network must learn new tasks without forgetting previously learned ones. EWC helps to mitigate catastrophic forgetting by regularizing the model’s weights.

The idea is to identify the important weights for previously learned tasks and then "protect" them while learning new tasks. Fisher Information plays a crucial role here: it is used to measure the importance of each weight for a given task.

### How EWC Works:

1. **Fisher Information Matrix**: For each task, EWC computes the Fisher Information, which indicates how much the loss will change if a particular weight is altered. Higher Fisher Information means that weight is more important for the task.
2. **Regularization Term**: During learning of new tasks, EWC adds a regularization term to the loss function. This term penalizes changes to important weights (those with high Fisher Information), preventing the model from forgetting what it learned on the previous tasks.
3. **Objective**: The goal is to allow the model to learn new tasks while preserving important weights from previous tasks. By using the Fisher Information, EWC ensures that the model focuses on adapting only the necessary weights for new tasks, while minimizing disruption to previously learned knowledge.

### Detailed Process:

1. **Train on Task 1 (data1)**: The model is initially trained on the first dataset (data1). During this training, the model learns to recognize entities specific to this dataset.
2. **Identify Important Parameters**: After training on data1, the model identifies which parameters (weights) are most important for the task. This is done by computing the Fisher Information Matrix (FIM), which measures the sensitivity of the loss function to changes in each parameter.
3. **Retain a Sample from Task 1**: A subset of data1 is stored to help the model remember the first task.
4. **Train on Task 2 (data2) with Task 1 Sample**: The model is then trained on the second dataset (data2). During this training, the model uses the stored sample from data1 and the FIM to ensure that important parameters from the first task are not significantly altered. This helps the model retain knowledge from the first task while learning the second task.
5. **Repeat for All Tasks**: The process is repeated for subsequent datasets (data3, data4, etc.). For each new task, the model uses the FIM and stored samples from previous tasks to maintain its performance on earlier tasks.

### Fisher Information Calculation Steps:
1. **Train the model on a dataset** and compute the gradients of the loss with respect to the model parameters.
2. **Estimate Fisher Information Matrix (FIM)** using these gradients, capturing parameter importance.
3. **Store learned weights** along with FIM values.
4. **When training on a new task**, add an EWC loss term that penalizes changes to important parameters based on their FIM scores.
5. **Iterate through tasks**, ensuring previous knowledge is preserved.

### Elastic Weight Consolidation (EWC) with Fisher Information

Elastic Weight Consolidation (EWC) helps prevent catastrophic forgetting in continual learning. The method works by regularizing the weights that are important for previously learned tasks using the Fisher Information.

#### Fisher Information

Fisher Information \( F_i \) measures the importance of each weight \( \theta_i \) to the learned task. It is computed as the expected value of the second derivative of the loss with respect to the weight:

$$
F_i = \mathbb{E} \left[ \left( \frac{\partial L}{\partial \theta_i} \right)^2 \right]
$$

In practice, Fisher Information is estimated using the diagonal of the Fisher Information Matrix, which can be approximated as:

$$
F_i = \frac{1}{N} \sum_{n=1}^{N} \left( \frac{\partial L_n}{\partial \theta_i} \right)^2
$$

where:
- \( L_n \) is the loss for the \( n \)-th training example,
- \( \theta_i \) is the \( i \)-th weight in the model,
- \( N \) is the number of training examples.

#### EWC Loss Function

The total loss function in EWC consists of two terms:
1. The loss for the new task.
2. A regularization term that penalizes large changes in important weights based on Fisher Information.

The EWC objective function is:

$$
L_{\text{total}} = L_{\text{new task}} + \lambda \sum_i \frac{F_i}{2} (\theta_i - \hat{\theta}_i)^2
$$

Where:
- \( L_{\text{new task}} \) is the loss function for the new task.
- \( \lambda \) is a regularization strength parameter that controls the importance of the EWC term.
- \( F_i \) is the Fisher Information for weight \( \theta_i \) related to the previous task.
- \( \hat{\theta}_i \) is the optimal value of weight \( \theta_i \) after training on the previous task.

### Key Idea

The goal of EWC is to allow the model to learn new tasks while preserving important weights from previous tasks by adding this regularization. The term \( (\theta_i - \hat{\theta}_i)^2 \) ensures that weights important for previous tasks don't change drastically, while the Fisher Information \( F_i \) quantifies how sensitive the loss is to changes in each weight.

## Experimentation and Performance Metrics

In this project, we have provided three datasets (`data1`, `data2`, `data3`) in the `data/` folder. These datasets are used to train a Named Entity Recognition (NER) model for medical named entity recognition. Each dataset corresponds to a different task:
- **T1**: Task 1 corresponding to `data1`
- **T2**: Task 2 corresponding to `data2`
- **T3**: Task 3 corresponding to `data3`

We have conducted experiments to compare the performance of continual learning against training on an aggregated dataset (T1+T2+T3). The results show that continual learning performs better than training on the combined dataset.

### Performance Metrics

| Entity           | T1              | T1 and T2       | T1, T2, and T3  | T1+T2+T3 Combined |
|------------------|-----------------|-----------------|-----------------|-------------------|
| allergy_name     | 0.738386        | 0.824524        | 0.901130        | 0.797745          |
| cancer           | 0.726179        | 0.793356        | 0.833593        | 0.742930          |
| chronic_disease  | 0.779630        | 0.805484        | 0.854898        | 0.783958          |
| treatment        | 0.777369        | 0.840783        | 0.881324        | 0.801621          |
| micro avg        | 0.769029        | 0.820854        | 0.865570        | 0.786875          |
| macro avg        | 0.755391        | 0.816037        | 0.867736        | 0.781564          |
| weighted avg     | 0.768511        | 0.820849        | 0.865853        | 0.786826          |

For more detailed experimentation and metrics, please refer to the `notebooks` directory(in the 'experiment' branch) where you can find Jupyter notebooks demonstrating the experiments.

Note: The provided datasets are for medical NER tasks, but you can replace them with any other datasets for different use cases.

## Running the Project

### Using Virtualenv

1. **Create and Activate Virtual Environment**:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate  # Windows
   ```
2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Project**:
   ```sh
   python -m src.main --data-dir data --output-dir output --wandb
   ```

### Using Docker

1. **Build Docker Image**:
   ```sh
   docker build -t ewc_ner .
   ```
2. **Run without Saving Output**:
   ```sh
   docker run -it --rm ewc_ner --data-dir /app/data --output-dir /app/output --wandb
   ```
3. **Run with Output Saved to Host**:
   ```sh
   docker run -it \
       -v /path/to/host/data:/app/data \
       -v /path/to/host/output:/app/output \
       ewc_ner
   ```

## Configuration

Modify `src/config.py` to adapt the model to different datasets and training setups. It is important to go through the config file and customize it according to your specific needs and data:

```python
@dataclass
class TrainingConfig:
    max_len: int
    train_batch_size: int
    valid_batch_size: int
    keep_batch_size: int
    epochs: int
    learning_rate: float
    max_grad_norm: float
    keep_sample_size: int # Number of data points to keep from Task N-1 (~Dataset N-1)
    train_size: float
    num_datasets: int  # Number of NER tasks (datasets)
    random_state: int
    drop_columns: list # Redundant columns to drop (from datasets)
    wandb_project_name: str

@dataclass
class ModelConfig:
    model_name: str # You can use any Encoder-Only Transformer, As it's TokenClassification
    num_labels: int
    device: str

ENTITY_SET = [] # The comprehensive list of all the labels possible
```

- **Modify `num_datasets`** to change the number of NER tasks.
- **Update `ENTITY_SET`** to redefine entity categories.
- **Preprocessing is configurable**, allowing adaptation to different data formats.
- **Datasets should be placed in the `data/` folder** and follow the naming convention: `data1`, `data2`, `data3`, etc. You can add as many datasets as needed, named sequentially.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
