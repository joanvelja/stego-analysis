from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import load_dataset

from utils.redactor import Redactor

GENDER_PATH = "rmcc11/lasr-toy-cv-gender"
GENDER_CORRELATION_PATH = "rmcc11/lasr-toy-cv-gender-correlations"
UNI_PATH = "rmcc11/lasr-toy-cv-with-paraphrased"

PATHS = {
    "gender": GENDER_PATH,
    "gender_correlation": GENDER_CORRELATION_PATH,
    "uni": UNI_PATH,
}


@dataclass
class ExperimentResult:
    model_name: str
    dataset_name: str
    is_redacted: bool
    accuracy: float
    precision: float
    recall: float
    f1: float
    timestamp: str
    fold: Optional[int] = None


class ExperimentLogger:
    def __init__(self, base_path: str = "./experiment_logs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.results: List[ExperimentResult] = []

    def log_result(
        self,
        model_name: str,
        dataset_name: str,
        is_redacted: bool,
        metrics: Dict,
        fold: Optional[int] = None,
    ):
        """Log a single experiment result"""
        result = ExperimentResult(
            model_name=model_name,
            dataset_name=dataset_name,
            is_redacted=is_redacted,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            timestamp=datetime.now().isoformat(),
            fold=fold,
        )
        self.results.append(result)

    def save_results(self):
        """Save results to disk"""
        results_df = pd.DataFrame([vars(r) for r in self.results])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(self.base_path / f"results_{timestamp}.csv", index=False)
        return results_df


class DataPrep:
    def __init__(self, dataset_name: str) -> None:
        self.paths = PATHS
        self.redactor = Redactor()
        self.clean_fn = (
            self.redactor.gender_neutralize
            if "gender" in dataset_name.lower()
            else self.redactor.clean_university
        )
        self.dataset_name = dataset_name
        print(f"Loading ... {self.dataset_name}")
        self.load_data()
        self.preprocess_data()

    def load_data(self) -> None:
        path = self.paths.get(self.dataset_name, None)
        if path is None:
            raise ValueError(
                "The dataset was not found. are you sure you typed it in correctly?"
            )

        dataset = load_dataset(path)
        # Keep only meaningful columns
        dataset = dataset.select_columns(["input", "label"])
        self.train = dataset["train"]
        self.test = dataset["test"]
        self.dataset = dataset

    def preprocess_data(self):
        train_dict = {
            "input": self.train["input"],
            "label": [
                1 if label.lower() == "pass" else 0 for label in self.train["label"]
            ],
        }

        test_dict = {
            "input": self.test["input"],
            "label": [
                1 if label.lower() == "pass" else 0 for label in self.test["label"]
            ],
        }

        self.train_df = pd.DataFrame(train_dict)
        self.test_df = pd.DataFrame(test_dict)

        self.train_df["input_redacted"] = self.train_df["input"].apply(self.clean_fn)
        self.test_df["input_redacted"] = self.test_df["input"].apply(self.clean_fn)
