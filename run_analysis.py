from data import DataPrep
from logistic_regression import LogisticRegressionWrapper


def main(dataset_name: str, redacted: bool):
    dp = DataPrep(dataset_name)
    lr = LogisticRegressionWrapper()
    lr.fit(dp.train_df, redacted=True)  # Should yield ~50% acc
    accuracy, report = lr.predict(dp.test_df)

    lr.reset()
    lr.fit(dp.train_df, redacted=False)  # Should yield 100% acc
    accuracy, report = lr.predict(dp.test_df)


if __name__ == "__main__":
    main("uni", True)
