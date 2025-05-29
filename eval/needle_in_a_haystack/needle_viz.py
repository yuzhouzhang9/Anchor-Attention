import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

sns.set_style("white")
sns.set_theme("poster", style="ticks", font_scale=1.2)
plt.rc("font", family="Times New Roman")
import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def plot_needle_viz(
    res_file,
    model_name,
    min_context=1000,
    max_context=100000,
    output_path="figures/",
    pattern="anchor_attn",
):
    def get_context_size(x, is_128k: bool = False):
        # if is_128k:
        return f"{round(x / 10000*10)}K"
        # if x > 990000:
        #     return f"{round(x / 1000000)}M"
        # if x <= 10000:
        #     return "10K" if x > 5000 else "1K"
        # if round(x / 1000) == 128:
        #     return "128K"
        # return f"{round(x / 10000)* 10}K"

    plt.rc("axes", titlesize=25)  # fontsize of the title
    plt.rc("axes", labelsize=25)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=20)  # fontsize of the x tick labels
    plt.rc("ytick", labelsize=20)  # fontsize of the y tick labels
    plt.rc("legend", fontsize=20)  # fontsize of the legend
    print(res_file)
    df = pd.read_json(res_file)
    accuracy_df = df.groupby(["context_length", "depth_percent"])["correct"].mean()
    accuracy_df = accuracy_df
    accuracy_df = accuracy_df.reset_index()
    accuracy_df = accuracy_df.rename(
        columns={
            "correct": "Score",
            "context_length": "Context Length",
            "depth_percent": "Document Depth",
        }
    )

    pivot_table = pd.pivot_table(
        accuracy_df,
        values="Score",
        index=["Document Depth", "Context Length"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(
        index="Document Depth", columns="Context Length", values="Score"
    )  # This will turn into a proper pivot

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
    )

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(14, 7))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        # cbar_kws={'label': 'Score'},
        vmin=0,
        vmax=1,
    )

    min_context_str = f"{min_context // 1000}K" if min_context >= 1000 else min_context
    max_context_str = f"{max_context // 1000}K" if max_context >= 1000 else max_context

    context = get_context_size(max_context)
    plt.title(
        f"{model_name} w/{pattern} {context} Context"
    )  # Adds a title
    plt.xlabel("Context Length")  # X-axis label
    plt.ylabel("Depth Percent (%)")  # Y-axis label

    # Centering x-ticks
    xtick_labels = pivot_table.columns.values
    xtick_labels = [get_context_size(x, context == "128K") for x in xtick_labels]
    print(xtick_labels)
    ax.set_xticks(np.arange(len(xtick_labels)) + 0.5, minor=False)
    ax.set_xticklabels(xtick_labels)

    # Drawing white grid lines
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(1)

    # Iterate over the number of pairs of gridlines you want
    for i in range(pivot_table.shape[0]):
        ax.axhline(i, color="white", lw=1)
    for i in range(pivot_table.shape[1]):
        ax.axvline(i, color="white", lw=1)

    # Ensure the ticks are horizontal and prevent overlap
    plt.xticks(rotation=60)
    plt.yticks(rotation=0)

    # Fit everything neatly into the figure area
    plt.tight_layout()

    print(f"model_name:{model_name}")
    print(f"pattern:{pattern}")
    # Save and Show the plot
    save_path = os.path.join(
        output_path,
        f"needle_viz_{model_name}_{pattern}_{min_context_str}_{max_context_str}.pdf",
    )
    # Ensure the directory exists before saving
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(save_path, dpi=1000)
    print(f"Needle plot saved to {save_path}.")
    plt.show()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--res_file", type=str, required=True)
    args.add_argument("--model_name", type=str, required=True)
    args.add_argument("--pattern", type=str)
    args.add_argument("--min_context", type=int,default=1000)
    args.add_argument("--max_context", type=int,default=100000)
    args = args.parse_args()

    plot_needle_viz(
        args.res_file,
        model_name=args.model_name,
        pattern=args.pattern,
        min_context=args.min_context,
        max_context=args.max_context,
    )
