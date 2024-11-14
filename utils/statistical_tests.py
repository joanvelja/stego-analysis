import matplotlib
import pandas as pd
from scipy import stats

matplotlib.use("TkAgg")  # Or "Qt5Agg", depending on your setup


class ResultsAnalyzer:
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df

    def compute_statistical_tests(self):
        """Compute statistical significance tests between redacted and non-redacted versions"""
        stats_results = []

        for model in self.results_df["model_name"].unique():
            for dataset in self.results_df["dataset_name"].unique():
                model_data = self.results_df[
                    (self.results_df["model_name"] == model)
                    & (self.results_df["dataset_name"] == dataset)
                ]

                if len(model_data) < 2:
                    continue

                redacted = model_data[model_data["is_redacted"]]["accuracy"]
                non_redacted = model_data[~model_data["is_redacted"]]["accuracy"]

                if len(redacted) > 0 and len(non_redacted) > 0:
                    t_stat, p_value = stats.ttest_ind(redacted, non_redacted)
                    stats_results.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                        }
                    )

        return pd.DataFrame(stats_results)

    def compute_binomial_tests(self):
        """Compute binomial tests for each experiment"""
        binomial_results = []

        for _, row in self.results_df.iterrows():
            n_samples = 2000
            n_correct = int(row["accuracy"] * n_samples)

            # Test against random chance (p=0.5 for binary classification)
            binom_test_random = stats.binomtest(
                n_correct, n_samples, p=0.5, alternative="greater"
            )

            result = {
                "model": row["model_name"],
                "dataset": row["dataset_name"],
                "is_redacted": row["is_redacted"],
                "accuracy": row["accuracy"],
                "n_correct": n_correct,
                "n_total": n_samples,
                "p_value_vs_random": binom_test_random.pvalue,
                "significant_vs_random": binom_test_random.pvalue < 0.05,
            }
            binomial_results.append(result)

        # Convert to DataFrame and add pairwise comparisons
        results_df = pd.DataFrame(binomial_results)

        # Add pairwise comparisons between redacted and non-redacted
        pairwise_results = []
        for model in results_df["model"].unique():
            for dataset in results_df["dataset"].unique():
                redacted = results_df[
                    (results_df["model"] == model)
                    & (results_df["dataset"] == dataset)
                    & results_df["is_redacted"]
                ]
                non_redacted = results_df[
                    (results_df["model"] == model)
                    & (results_df["dataset"] == dataset)
                    & ~results_df["is_redacted"]
                ]

                if len(redacted) > 0 and len(non_redacted) > 0:
                    # Compare redacted vs non-redacted using binomial test
                    binom_test_pair = stats.binomtest(
                        int(redacted["n_correct"].iloc[0]),
                        redacted["n_total"].iloc[0],
                        p=non_redacted["accuracy"].iloc[0],
                        alternative="two-sided",
                    )

                    pairwise_results.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "redacted_accuracy": redacted["accuracy"].iloc[0],
                            "non_redacted_accuracy": non_redacted["accuracy"].iloc[0],
                            "p_value_redacted_vs_non": binom_test_pair.pvalue,
                            "significant_difference": binom_test_pair.pvalue < 0.05,
                        }
                    )

        return results_df, pd.DataFrame(pairwise_results)

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        summary = (
            self.results_df.groupby(["model_name", "dataset_name", "is_redacted"])
            .agg(
                {
                    "accuracy": ["mean", "std"],
                    "precision": ["mean", "std"],
                    "recall": ["mean", "std"],
                    "f1": ["mean", "std"],
                }
            )
            .round(4)
        )

        return summary
