import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ====== Combined Model Scores (New Data) ======
combined_df = pd.DataFrame(
    [
        ["XGBoost", 0.9981, 0.9933, 22.6207, 22.6207, 9.6930, 9.6930],
        ["RF", 0.9963, 0.9933, 22.7748, 22.7748, 9.8764, 9.8764],
        ["LSTM", 0.9876, 0.9873, 31.2977, 31.2977, 20.1867, 20.1867],
        ["CNN-LSTM", 0.9891, 0.9896, 28.2191, 28.2191, 17.0076, 17.0076],
        ["TFT", 0.9959, 0.9472, 63.6943, 63.6943, 28.5793, 28.5793],
        ["TCN", 0.9936, 0.9461, 64.3800, 64.3800, 28.8272, 28.8272],
    ],
    columns=[
        "Model",
        "Train R²",
        "Test R²",
        "Train RMSE",
        "Test RMSE",
        "Train MAE",
        "Test MAE",
    ],
)

# ====== Metric Mapping ======
metrics = {
    "R²": ["Train R²", "Test R²"],
    "RMSE": ["Train RMSE", "Test RMSE"],
    "MAE": ["Train MAE", "Test MAE"],
}

# ====== Global Plot Settings ======
plt.rcParams.update({"font.family": "serif", "font.size": 12})

palette = {"Train": "#FF6347", "Test": "#32CD32"}  # Red & Green

# ====== Plotting ======
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
axes = axes.flatten()

for ax, (metric_name, cols) in zip(axes, metrics.items()):
    df_melted = combined_df.melt(
        id_vars=["Model"], value_vars=cols, var_name="DataType", value_name=metric_name
    )
    df_melted["Type"] = df_melted["DataType"].str.split(" ").str[0]  # Train/Test

    sns.barplot(
        data=df_melted, x="Model", y=metric_name, hue="Type", palette=palette, ax=ax
    )

    ax.set_title(
        f"{metric_name} by Model (Train vs Test)", fontsize=16, fontweight="bold"
    )
    ax.set_ylabel(metric_name, fontsize=14, fontweight="bold")
    ax.set_xlabel("", fontsize=14, fontweight="bold")

    ax.tick_params(axis="x", labelsize=11, width=2)
    ax.tick_params(axis="y", labelsize=12, width=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontweight="bold")
    ax.set_yticklabels(ax.get_yticks(), fontweight="bold")

    ax.legend_.remove()

    # Set y-axis upper limit for R² to 1.0
    if metric_name == "R²":
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))


# ====== Common Legend ======
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Data Type",
    loc="upper right",
    ncol=2,
    frameon=False,
    fontsize=14,
    title_fontsize=14,
)

# ====== Main Title ======
fig.suptitle("Model Performance Comparison", fontsize=20, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("Model_Performance_Comparison.png", dpi=300, bbox_inches="tight")
plt.show()
