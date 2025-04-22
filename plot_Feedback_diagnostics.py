import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')

# read the CSV (skipping comments)
df = pd.read_csv("feedback_diagnostics.csv", comment="#",
                 names=["delta_u","rel_inc","r","n_ngb","h_star","E_ratio"])

# split neighbor vs star records
neighbors = df.dropna(subset=["delta_u"])
stars     = df.dropna(subset=["n_ngb"])

# create 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 1) Histogram of delta_u (log-log)
axes[0].hist(neighbors["delta_u"], bins=50, log=True)
axes[0].set_xscale("log")
axes[0].set_xlabel("Δu (post-feedback specific internal energy)")
axes[0].set_ylabel("Count")
axes[0].set_title("Δ Thermal Energy Injection After Feedback")

# 2) rel_inc vs radius (scatter)
axes[1].scatter(neighbors["r"], neighbors["rel_inc"], s=2, alpha=0.5)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Distance r [kpc]")
axes[1].set_ylabel("Δu / u_before")
axes[1].set_title("Fractional Thermal Energy Increase vs Distance")

# 3) Histogram of neighbor-counts
axes[2].hist(stars["n_ngb"], bins=30)
axes[2].set_xlabel("Neighbors per Star")
axes[2].set_ylabel("Events")
axes[2].set_title("Number of Neighbors Receiving Feedback")

# 4) Histogram of h_star
axes[3].hist(stars["h_star"], bins=30)
axes[3].set_xlabel("Kernel Radius h [kpc]")
axes[3].set_ylabel("Events")
axes[3].set_title("Feedback Kernel Radii Distribution")

# 5) Energy ratio
axes[4].hist(stars["E_ratio"], bins=30)
axes[4].set_xlabel("E_applied / E_input")
axes[4].set_ylabel("Counts")
axes[4].set_title("Energy Conservation Ratio per Star Event")

# hide the empty sixth subplot
axes[5].axis("off")

plt.tight_layout()
plt.show()
