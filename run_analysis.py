import os
import random

import numpy as np
import pandas as pd
import torch

from bert_classifier import BertClassifier
from data import DataPrep, ExperimentLogger
from utils.plots import plot_results


# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(dataset_name: str, seed: int):
    """Runs the experiment for a single seed and returns the result metrics."""
    set_seeds(seed)
    logger = ExperimentLogger()
    dp = DataPrep(dataset_name)

    # BERT experiments
    bc = BertClassifier()

    # Non-redacted
    bc.train(dp.train_df, dp.test_df, redacted=False)
    metrics_non_redacted = bc.evaluate()
    fixed_metrics_non_redacted = {
        k.split("_", maxsplit=1)[1]: v
        for k, v in metrics_non_redacted.items()
        if "_" in k
    }
    logger.log_result(
        model_name="BERT",
        dataset_name=dataset_name,
        is_redacted=False,
        metrics=fixed_metrics_non_redacted,
    )

    # Redacted
    bc.reset()
    bc.train(dp.train_df, dp.test_df, redacted=True)
    metrics_redacted = bc.evaluate()
    fixed_metrics_redacted = {
        k.split("_", maxsplit=1)[1]: v for k, v in metrics_redacted.items() if "_" in k
    }
    logger.log_result(
        model_name="BERT",
        dataset_name=dataset_name,
        is_redacted=True,
        metrics=fixed_metrics_redacted,
    )

    # Save and return results
    results_df = logger.save_results()
    return results_df


def average_results_across_seeds(dataset_name: str, n_seeds: int):
    """Run experiments across multiple seeds and calculate average metrics."""
    all_results = []

    # Run the experiment for each seed and store the results
    for seed in range(n_seeds):
        results_df = run_experiment(dataset_name, seed)
        all_results.append(results_df)

    # Concatenate results and calculate mean metrics
    combined_results = pd.concat(all_results)
    avg_results = (
        combined_results.groupby(["model_name", "dataset_name", "is_redacted"])
        .mean()
        .reset_index()
    )

    # Save the averaged results
    avg_results.to_csv(
        f"./experiment_logs/{dataset_name}/averaged_results.csv", index=False
    )
    plot_results(
        avg_results, f"./experiment_logs/{dataset_name}/averaged_results_plot.png"
    )

    return avg_results


def main():
    for dataset_name in ["gender", "gender_correlation", "uni"]:
        os.makedirs(f"./experiment_logs/{dataset_name}", exist_ok=True)

        # Set number of seeds to average over
        n_seeds = 10

        # Run and average results across seeds
        avg_results = average_results_across_seeds(dataset_name, n_seeds)

        print(f"\nAveraged Results for dataset: {dataset_name}")
        print(avg_results)


if __name__ == "__main__":
    main()
