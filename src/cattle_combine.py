import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import warnings
import numpy as np
from scipy import stats
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--pathTarget", type=str, help="Path to the directory where data is going to be stored.")
parser.add_argument("--pathHrsAbove", type=str, help="Path to the hours above threshold dataset.")
parser.add_argument('--pathCattle', type=str, help='Path to the cattle distribution dataset.')
args=parser.parse_args()

path_target = args.pathTarget
os.makedirs(path_target, exist_ok=True)
os.chdir(path_target)


def convert_longitude(ds, lon="longitude"):
    """
    Convert longitude from 0-360 to -180 to 180 format in an xarray dataset

    Args:
        ds (xarray.Dataset): Input dataset with longitude coordinates

    Returns:
        xarray.Dataset: Dataset with converted longitude coordinates
    """
    # Select longitudes > 180 and subtract 360 to convert to negative values
    ds.coords[lon] = (ds.coords[lon] + 180) % 360 - 180

    # Sort the dataset by longitude
    return ds.sortby(lon)


def create_mask(ds, var, low_lim=0, up_lim=6):
    return xr.where((ds[var] >= low_lim) & (ds[var] < up_lim), 1, 0)


# Read the hours above threshold dataset
ds = xr.open_dataset(args.pathHrsAbove).load()

ds = convert_longitude(ds, lon="lon")
# Add the cattle distribution
ds_distr = xr.open_dataset(args.pathCattle)
ds = ds.merge(ds_distr)

# Create boolean masks for the hours of the day above
# the threshold variable as follows:
# 0-6 hrs: 1
# 6-12 hrs: 2
# 12-18 hrs: 3
# 18-24 hrs: 4
for c_, lims in list(zip([1, 2, 3, 4], [[0, 6], [6, 12], [12, 18], [18, 24]])):
    ds[f"class_{c_}"] = ds.cattle * create_mask(
        ds=ds, var="THI", low_lim=lims[0], up_lim=lims[1]
    )
    gc.collect()

# Calculate the total cattle worldwide
total_cattle = ds_distr.sum(dim=["lat", "lon"]).cattle.item()


# Calculate the yearly total cattle at risk and convert to dataframe
df = (
    ds[["class_1", "class_2", "class_3", "class_4"]]
    .sum(dim=["lon", "lat"])
    .to_dataframe()
    .reset_index(drop=False)
)
gc.collect()

# Express the cattle in each class as a percentage of the total cattle worldwide
df = df.assign(
    class_1=100 * (df.class_1 / total_cattle),
    class_2=100 * (df.class_2 / total_cattle),
    class_3=100 * (df.class_3 / total_cattle),
    class_4=100 * (df.class_4 / total_cattle),
)

# Plot the timelines for the 68.8 threshold
sns.set(font_scale=2.3)
sns.set_style("whitegrid")

fig, ax = plt.subplots(1, 2, figsize=(22, 10), sharey=True)

for c_, label in list(zip([1, 2, 3, 4], ["0-25%", "25-50%", "50-75%", "75-100%"])):
    sns.lineplot(
        data=df.loc[(df.threshold == 68.8) & (df.scenario == "ssp245")],
        x="time",
        y=f"class_{c_}",
        linewidth=3,
        linestyle="-.",
        ax=ax[0],
        label=label,
        errorbar=("ci", 95),
    )
    sns.lineplot(
        data=df.loc[(df.threshold == 68.8) & (df.scenario == "ssp585")],
        x="time",
        y=f"class_{c_}",
        linewidth=3,
        linestyle="-.",
        ax=ax[1],
        label=label,
        errorbar=("ci", 95),
    )
ax[0].set(xlabel="Time", ylabel="Cattle at risk ($\%$ of total)", title="SSP2-45")
ax[1].set(xlabel="Time", ylabel="Cattle at risk ($\%$ of total)", title="SSP5-85")
ax[0].legend("", frameon=False)
ax[1].legend(frameon=False)
plt.tight_layout()
plt.savefig("Cattle_at_risk.png", dpi=150)
plt.close()


# ------------------------ #
# Plot the timelines for the 68.8 threshold
sns.set(font_scale=2.3)
sns.set_style("whitegrid")

fig, ax = plt.subplots(1, 2, figsize=(22, 10), sharey=True)

# Define colors beforehand to ensure consistency
colors = sns.color_palette()[:4]

for idx, (c_, label) in enumerate(
    list(zip([1, 2, 3, 4], ["0-25%", "25-50%", "50-75%", "75-100%"]))
):
    # SSP2-45
    data_245 = df.loc[(df.threshold == 68.8) & (df.scenario == "ssp245")]
    sns.lineplot(
        data=data_245,
        x="time",
        y=f"class_{c_}",
        linewidth=3,
        linestyle="-",
        ax=ax[0],
        label=label,
        errorbar=("ci", 95),
        color=colors[idx],
    )

    # Add linear regression for SSP2-45
    x_245 = data_245["time"].astype(np.int64)
    y_245 = data_245[f"class_{c_}"]
    slope_245, intercept_245, _, _, _ = stats.linregress(x_245, y_245)
    line_245 = slope_245 * x_245 + intercept_245
    ax[0].plot(data_245["time"], line_245, "--", color=colors[idx], linewidth=2)

    # SSP5-85
    data_585 = df.loc[(df.threshold == 68.8) & (df.scenario == "ssp585")]
    sns.lineplot(
        data=data_585,
        x="time",
        y=f"class_{c_}",
        linewidth=3,
        linestyle="-",
        ax=ax[1],
        label=label,
        errorbar=("ci", 95),
        color=colors[idx],
    )

    # Add linear regression for SSP5-85
    x_585 = data_585["time"].astype(np.int64)
    y_585 = data_585[f"class_{c_}"]
    slope_585, intercept_585, _, _, _ = stats.linregress(x_585, y_585)
    line_585 = slope_585 * x_585 + intercept_585
    ax[1].plot(data_585["time"], line_585, "--", color=colors[idx], linewidth=2)

ax[0].set(xlabel="Time", ylabel="Cattle at risk ($\%$ of total)", title="SSP2-45")
ax[1].set(xlabel="Time", ylabel="Cattle at risk ($\%$ of total)", title="SSP5-85")
ax[0].legend("", frameon=False)
ax[1].legend(frameon=False)
plt.tight_layout()
sns.despine(bottom=True, right=True, left=True, top=True)
plt.savefig("Cattle_at_risk.png", dpi=150)
plt.close()


# Plot the timelines for the 84 threshold
sns.set(font_scale=2.3)
sns.set_style("whitegrid")

fig, ax = plt.subplots(1, 2, figsize=(22, 10), sharey=True)

# Define colors beforehand to ensure consistency
colors = sns.color_palette()[:4]

for idx, (c_, label) in enumerate(
    list(zip([1, 2, 3, 4], ["0-25%", "25-50%", "50-75%", "75-100%"]))
):
    # SSP2-45
    data_245 = df.loc[(df.threshold == 84) & (df.scenario == "ssp245")]
    sns.lineplot(
        data=data_245,
        x="time",
        y=f"class_{c_}",
        linewidth=3,
        linestyle="-",
        ax=ax[0],
        label=label,
        errorbar=("ci", 95),
        color=colors[idx],
    )

    # Add linear regression for SSP2-45
    x_245 = data_245["time"].astype(np.int64)
    y_245 = data_245[f"class_{c_}"]
    slope_245, intercept_245, _, _, _ = stats.linregress(x_245, y_245)
    line_245 = slope_245 * x_245 + intercept_245
    ax[0].plot(data_245["time"], line_245, "--", color=colors[idx], linewidth=2)

    # SSP5-85
    data_585 = df.loc[(df.threshold == 84) & (df.scenario == "ssp585")]
    sns.lineplot(
        data=data_585,
        x="time",
        y=f"class_{c_}",
        linewidth=3,
        linestyle="-",
        ax=ax[1],
        label=label,
        errorbar=("ci", 95),
        color=colors[idx],
    )

    # Add linear regression for SSP5-85
    x_585 = data_585["time"].astype(np.int64)
    y_585 = data_585[f"class_{c_}"]
    slope_585, intercept_585, _, _, _ = stats.linregress(x_585, y_585)
    line_585 = slope_585 * x_585 + intercept_585
    ax[1].plot(data_585["time"], line_585, "--", color=colors[idx], linewidth=2)

ax[0].set(xlabel="Time", ylabel="Cattle at risk ($\%$ of total)", title="SSP2-45")
ax[1].set(xlabel="Time", ylabel="Cattle at risk ($\%$ of total)", title="SSP5-85")
ax[0].legend("", frameon=False)
ax[1].legend(frameon=False)
plt.tight_layout()
sns.despine(bottom=True, right=True, left=True, top=True)
plt.savefig("Cattle_at_risk_84.png", dpi=150)
plt.close()


# ================== COMBINED ======================== #
# Set up the plot style
sns.set(font_scale=2.3)
sns.set_style("whitegrid")

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(22, 20), sharey=True, sharex=True)

# Define colors beforehand to ensure consistency
colors = sns.color_palette()[:4]


# Function to plot for a specific threshold and row
def plot_threshold(threshold, row_idx):
    for idx, (c_, label) in enumerate(
        list(zip([1, 2, 3, 4], ["0-25%", "25-50%", "50-75%", "75-100%"]))
    ):
        # SSP2-45
        data_245 = df.loc[(df.threshold == threshold) & (df.scenario == "ssp245")]
        sns.lineplot(
            data=data_245,
            x="time",
            y=f"class_{c_}",
            linewidth=3,
            linestyle="-",
            ax=axes[row_idx, 0],
            label=label,
            errorbar=("ci", 95),
            color=colors[idx],
        )

        # Add linear regression for SSP2-45
        x_245 = data_245["time"].astype(np.int64)
        y_245 = data_245[f"class_{c_}"]
        slope_245, intercept_245, _, _, _ = stats.linregress(x_245, y_245)
        line_245 = slope_245 * x_245 + intercept_245
        axes[row_idx, 0].plot(
            data_245["time"], line_245, "--", color=colors[idx], linewidth=2
        )

        # SSP5-85
        data_585 = df.loc[(df.threshold == threshold) & (df.scenario == "ssp585")]
        sns.lineplot(
            data=data_585,
            x="time",
            y=f"class_{c_}",
            linewidth=3,
            linestyle="-",
            ax=axes[row_idx, 1],
            label=label,
            errorbar=("ci", 95),
            color=colors[idx],
        )

        # Add linear regression for SSP5-85
        x_585 = data_585["time"].astype(np.int64)
        y_585 = data_585[f"class_{c_}"]
        slope_585, intercept_585, _, _, _ = stats.linregress(x_585, y_585)
        line_585 = slope_585 * x_585 + intercept_585
        axes[row_idx, 1].plot(
            data_585["time"], line_585, "--", color=colors[idx], linewidth=2
        )


# Plot for threshold 68.8 (top row)
plot_threshold(68.8, 0)
# Plot for threshold 84 (bottom row)
plot_threshold(84, 1)

# Set titles and labels
axes[0, 0].set(
    xlabel="", ylabel="Cattle at risk ($\%$ of total)", title="SSP2-45 (THI > 68.8)"
)
axes[0, 1].set(xlabel="", ylabel="", title="SSP5-85 (THI > 68.8)")
axes[1, 0].set(
    xlabel="Time", ylabel="Cattle at risk ($\%$ of total)", title="SSP2-45 (THI > 84.0)"
)
axes[1, 1].set(xlabel="Time", ylabel="", title="SSP5-85 (THI > 84.0)")

# Handle legends
axes[0, 0].legend("", frameon=False)
axes[0, 1].legend("", frameon=False)
axes[1, 0].legend("", frameon=False)
axes[1, 1].legend(frameon=False)

plt.tight_layout()
sns.despine(bottom=True, right=True, left=True, top=True)
plt.savefig("Cattle_at_risk_combined.png", dpi=150)
plt.close()
