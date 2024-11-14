from typing import Dict, Tuple

from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class LogisticRegressionWrapper:
    def __init__(self, redacted: bool = False) -> None:
        # Vectorize text using simple CountVectorizer
        self.vectorizer = CountVectorizer()
        self.redacted = redacted

        self.key = "input_redacted" if self.redacted else "input"

    def prep_train(self, data) -> None:
        self.X_train_vect = self.vectorizer.fit_transform(self.train_df["key"])

    def prep_test(self, data) -> None:
        self.X_test_vect = self.vectorizer.transform(self.test_df["key"])

    def fit(self, train_df: DataFrame) -> None:
        self.train_df = train_df

        # Vectorize data for LR
        self.prep_train(self.train_df)

        # Train a Logistic Regression model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train_vect, self.train_df["label"])

    def predict(
        self, test_df: DataFrame, eval: bool = True
    ) -> Tuple[float, Dict] | None:
        # Vectorize data for prediction
        self.test_df = test_df
        self.prep_test(self.prep_test)

        # Make predictions on test data
        y_pred = self.model.predict(self.X_test_vect)

        if not eval:
            return

        # Evaluate the model
        accuracy = accuracy_score(test_df["label"], y_pred)
        report = classification_report(test_df["label"], y_pred, output_dict=True)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(report)

        return accuracy, report
