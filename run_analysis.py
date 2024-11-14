from bert_classifier import BertClassifier
from data import DataPrep
from logistic_regression import LogisticRegressionWrapper


def main(dataset_name: str):
    dp = DataPrep(dataset_name)
    lr = LogisticRegressionWrapper()
    lr.fit(dp.train_df, redacted=True)  # Should yield ~50% acc
    accuracy, report = lr.predict(dp.test_df)

    lr.reset()
    lr.fit(dp.train_df, redacted=False)  # Should yield 100% acc
    accuracy, report = lr.predict(dp.test_df)

    bc = BertClassifier()
    bc.train(dp.train_df, dp.test_df, redacted=False)
    bc.evaluate()
    bc.reset()

    bc.train(dp.train_df, dp.test_df, redacted=True)
    bc.evaluate()


if __name__ == "__main__":
    main("uni")
