import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')

# Read the CSV (skipping comments)
df = pd.read_csv(
    "feedback_diagnostics.csv",
    comment="#",
    names=["delta_u", "delta_v", "rel_inc", "r", "n_ngb", "h_star", "E_ratio"]
)

# Split neighbor vs. star records
neighbors = df.dropna(subset=["delta_u"])
stars     = df.dropna(subset=["n_ngb"])

# Create 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 1) Histogram of Δu (thermal energy kicks)
axes[0].hist(neighbors["delta_u"], bins=50, log=True)
axes[0].set_xscale("log")
axes[0].set_xlabel("Δu (post-feedback specific internal energy)")
axes[0].set_ylabel("Count")
axes[0].set_title("Δ Thermal Energy Injection After Feedback")

# 2) Histogram of Δv (velocity kicks)
axes[1].hist(neighbors["delta_v"], bins=50, log=True)
axes[1].set_xscale("log")
axes[1].set_xlabel("Δv (feedback velocity kick)")
axes[1].set_ylabel("Count")
axes[1].set_title("Velocity Kick Distribution After Feedback")

# 3) Relative energy increase vs. distance
axes[2].scatter(neighbors["r"], neighbors["rel_inc"], s=2, alpha=0.5)
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_xlabel("Distance r [kpc]")
axes[2].set_ylabel("Δu / u_before")
axes[2].set_title("Fractional Thermal Energy Increase vs Distance")

# 4) Histogram of neighbor-counts
axes[3].hist(stars["n_ngb"], bins=30)
axes[3].set_xlabel("Neighbors per Star")
axes[3].set_ylabel("Events")
axes[3].set_title("Number of Neighbors Receiving Feedback")

# 5) Histogram of feedback kernel radii
axes[4].hist(stars["h_star"], bins=30)
axes[4].set_xlabel("Kernel Radius h [kpc]")
axes[4].set_ylabel("Events")
axes[4].set_title("Feedback Kernel Radii Distribution")

# 6) Energy conservation ratio per star
axes[5].hist(stars["E_ratio"], bins=30)
axes[5].set_xlabel("E_applied / E_input")
axes[5].set_ylabel("Counts")
axes[5].set_title("Energy Conservation Ratio per Star Event")

plt.tight_layout()
plt.show()
