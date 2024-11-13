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


class DataPrep:
    def __init__(self, dataset_name: str) -> None:
        self.paths = PATHS
        self.redactor = Redactor()
        self.clean_fn = self.redactor.clean(dataset_name)
        self.dataset_name = dataset_name
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
