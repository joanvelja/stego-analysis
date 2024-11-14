from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns


def plot_results(results_df, output_path: Optional[str] = None):
    """Generate visualizations of the results, faceted by model with improved UX."""
    # Set plot style
    sns.set_style("whitegrid")
    sns.set_palette("coolwarm")

    # Convert 'is_redacted' to categorical for clearer labels
    results_df["Redaction Status"] = results_df["is_redacted"].map(
        {True: "Redacted", False: "Non-Redacted"}
    )

    # Set up a FacetGrid for scatter plots, with each model as a separate subplot
    g = sns.FacetGrid(
        data=results_df, col="model_name", height=6, aspect=1, sharey=True
    )

    # Map scatter plots to each facet, with custom settings
    g.map_dataframe(
        sns.scatterplot,
        x="dataset_name",
        y="accuracy",
        hue="Redaction Status",
        style="Redaction Status",
        palette="Set1",
        s=120,
        edgecolor="k",
    )

    # Customize plot titles and labels
    g.fig.suptitle(
        "Model Accuracy by Dataset: Redacted vs Non-Redacted", fontsize=16, y=1.02
    )
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Dataset", "Accuracy")

    # Rotate x-axis labels for readability
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Adjust legend settings
    g.add_legend(title="Redaction Status")
    for legend_handle in g._legend.legendHandles:
        legend_handle.set_edgecolor("k")

    # Make layout adjustments and show/save plot
    plt.tight_layout()
    if output_path:
        g.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
