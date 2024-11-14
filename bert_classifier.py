import gc
import os

import torch
from pandas import DataFrame
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BERT_MODEL = "distilbert/distilbert-base-uncased"


# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Use clone().detach() for proper tensor construction
        item = {
            key: val[idx].clone().detach()
            if isinstance(val[idx], torch.Tensor)
            else torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx].tolist())
        return item

    def __len__(self):
        return len(self.labels)


class BertClassifier:
    def __init__(self) -> None:
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Load BERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                BERT_MODEL,
                num_labels=2,
            )
            .to(self.device)
            .train()
        )

    def prepare_data(
        self,
    ):
        self.train_encodings = self.tokenizer(
            list(self.train_df[self.key]),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        self.test_encodings = self.tokenizer(
            list(self.test_df[self.key]),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        self.train_dataset = TextDataset(self.train_encodings, self.train_df["label"])
        self.test_dataset = TextDataset(self.test_encodings, self.test_df["label"])

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        report = classification_report(
            labels, preds, target_names=["FAIL", "PASS"], output_dict=True
        )
        return {
            "accuracy": accuracy,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
        }

    def train(self, train_df: DataFrame, test_df: DataFrame, redacted: bool = False):
        self.train_df = train_df
        self.test_df = test_df
        self.key = "input_redacted" if redacted else "input"
        self.prepare_data()

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="steps",
            max_steps=1200,
            per_device_train_batch_size=4,
            bf16=True,  # Enable mixed precision
            logging_dir="./logs",  # Directory for storing logs
            logging_steps=300,  # Log every 250 steps
            save_steps=1200,  # Save checkpoint every 500 steps
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_strategy="steps",
            gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
        )

        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            optimizers=(optimizer, None),  # None for scheduler
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()

    def reset(self):
        """
        Saves the current model state, cleans up memory, and reinitializes the classifier.
        This is useful for training multiple iterations or when memory needs to be freed.
        """

        # Create directory if it doesn't exist
        os.makedirs("model_checkpoints", exist_ok=True)

        # Save current model state if it exists and has been trained
        if hasattr(self, "trainer") and self.trainer is not None:
            checkpoint_path = os.path.join("model_checkpoints", "last_model_state")
            self.trainer.save_model(checkpoint_path)

            # Clean up trainer
            del self.trainer
            self.trainer = None

        # Clean up datasets
        if hasattr(self, "train_dataset"):
            del self.train_dataset
        if hasattr(self, "test_dataset"):
            del self.test_dataset

        # Clean up encodings
        if hasattr(self, "train_encodings"):
            del self.train_encodings
        if hasattr(self, "test_encodings"):
            del self.test_encodings

        # Clean up dataframes
        if hasattr(self, "train_df"):
            del self.train_df
        if hasattr(self, "test_df"):
            del self.test_df

        # Move model to CPU before deletion if it's on GPU
        if hasattr(self, "model"):
            self.model.cpu()
            del self.model

        # Clean up tokenizer
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        # Reinitialize the classifier
        self.__init__()
