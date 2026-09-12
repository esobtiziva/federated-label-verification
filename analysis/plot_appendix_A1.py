import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Appendix Figure A1
# Seed-wise effect of label corruption on R-squared
# ---------------------------------------------------------

# Read final five-seed paired-effect data
file_name = "paired_effects_all_five_seeds.csv"
df = pd.read_csv(file_name)

# Print columns so that the source structure is documented
print("Columns in input file:")
print(df.columns.tolist())

# ---------------------------------------------------------
# Automatically identify the required columns
# ---------------------------------------------------------

def find_column(possible_names):
    """Return the first matching column name."""
    lower_map = {c.lower().strip(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


partition_col = find_column([
    "partition", "partitioning", "partition_type",
    "partition_strategy", "data_partition"
])

client_col = find_column([
    "clients", "num_clients", "n_clients",
    "client_count"
])

gamma_col = find_column([
    "gamma", "noise_gamma", "noise_level"
])

seed_col = find_column([
    "seed", "random_seed"
])

delta_col = find_column([
    "delta_r2_noise",
    "noise_delta_r2",
    "delta_noise_r2",
    "r2_noise_delta",
    "delta_r2_baseline_noise"
])

# If the paired-difference column is not already present,
# calculate it from baseline and noise R2.
if delta_col is None:

    baseline_col = find_column([
        "baseline_r2", "r2_baseline"
    ])

    noise_col = find_column([
        "noise_r2", "r2_noise"
    ])

    if baseline_col is None or noise_col is None:
        raise ValueError(
            "Could not identify either the paired noise-effect "
            "column or the Baseline/Noise R2 columns."
        )

    df["delta_r2_noise_plot"] = (
        df[noise_col] - df[baseline_col]
    )

    delta_col = "delta_r2_noise_plot"


# Check that all structural columns were found
required = {
    "partition": partition_col,
    "clients": client_col,
    "gamma": gamma_col,
    "seed": seed_col,
    "delta R2": delta_col
}

for name, column in required.items():
    if column is None:
        raise ValueError(
            f"Could not identify the {name} column."
        )

print("\nColumns used:")
for name, column in required.items():
    print(f"{name}: {column}")


# ---------------------------------------------------------
# Standardise configuration labels
# ---------------------------------------------------------

def make_configuration(row):

    partition = str(row[partition_col]).strip().lower()
    clients = int(row[client_col])

    if "non" in partition:
        prefix = "non-IID"
    else:
        prefix = "IID"

    return f"{prefix}-{clients}"


df["Configuration"] = df.apply(
    make_configuration, axis=1
)

config_order = [
    "IID-2",
    "IID-5",
    "non-IID-2",
    "non-IID-5"
]


# ---------------------------------------------------------
# Check the data before plotting
# ---------------------------------------------------------

print("\nSeed-wise label-noise effects:")
print(
    df[
        [
            seed_col,
            gamma_col,
            "Configuration",
            delta_col
        ]
    ].sort_values(
        [gamma_col, "Configuration", seed_col]
    ).to_string(index=False)
)


# ---------------------------------------------------------
# Create Figure A1
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5.5),
    sharey=True
)

gamma_values = [0.5, 1.0]

# Small horizontal offsets make the five seed points visible
offsets = np.linspace(-0.16, 0.16, 5)

for ax, gamma_value in zip(axes, gamma_values):

    subset_gamma = df[
        np.isclose(
            pd.to_numeric(df[gamma_col]),
            gamma_value
        )
    ].copy()

    for x, config in enumerate(config_order):

        temp = subset_gamma[
            subset_gamma["Configuration"] == config
        ].sort_values(seed_col)

        values = temp[delta_col].astype(float).values
        seeds = temp[seed_col].values

        if len(values) == 0:
            continue

        # Individual seed observations
        current_offsets = offsets[:len(values)]

        ax.scatter(
            np.full(len(values), x) + current_offsets,
            values,
            s=55,
            zorder=3
        )

        

        # Mean value
        mean_value = np.mean(values)

        ax.scatter(
            x,
            mean_value,
            marker="D",
            s=85,
            edgecolors="black",
            zorder=4,
            label="Mean" if x == 0 else None
        )

    # Zero reference line
    ax.axhline(
        0,
        linewidth=1.2,
        linestyle="--"
    )

    ax.set_xticks(range(len(config_order)))
    ax.set_xticklabels(config_order)

    ax.set_title(
        rf"Noise severity: $\gamma={gamma_value}$"
    )

    ax.set_xlabel("Federated configuration")

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.4
    )


axes[0].set_ylabel(
    r"Paired change in $R^2$ "
    r"($R^2_{\mathrm{Noise}} - R^2_{\mathrm{Baseline}}$)"
)

# Overall title
fig.suptitle(
    "Seed-wise Effect of Label Corruption on Predictive Performance",
    fontsize=14
)

# Explanation at bottom
fig.text(
    0.5,
    0.01,
    "Points represent individual random seeds; diamonds represent "
    "the five-seed mean. Values below zero indicate performance "
    "degradation following label corruption.",
    ha="center",
    fontsize=9
)

plt.tight_layout(
    rect=[0, 0.06, 1, 0.94]
)

# ---------------------------------------------------------
# Save publication-quality versions
# ---------------------------------------------------------

plt.savefig(
    "Figure_A1_Label_Noise_Effect.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "Figure_A1_Label_Noise_Effect.pdf",
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as:\n"
    "Figure_A1_Label_Noise_Effect.png\n"
    "Figure_A1_Label_Noise_Effect.pdf"
)