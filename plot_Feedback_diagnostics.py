import pandas as pd
import matplotlib.pyplot as plt

# read the CSV (skipping comments)
df = pd.read_csv("feedback_diagnostics.csv", comment="#",
                 names=["delta_u","rel_inc","r","n_ngb","h_star","E_ratio"])

# split neighbor vs star records
neighbors = df.dropna(subset=["delta_u"])
stars     = df.dropna(subset=["n_ngb"])

# 1) Histogram of delta_u
plt.figure()
plt.hist(neighbors["delta_u"], bins=50, log=True)
plt.xscale("log")
plt.xlabel("delta_u")
plt.ylabel("Count")
plt.title("delta_u distribution")
plt.tight_layout()

# 2) rel_increase vs radius
plt.figure()
plt.scatter(neighbors["r"], neighbors["rel_inc"], s=2, alpha=0.5)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("r [kpc]")
plt.ylabel("delta_u / u_before")
plt.title("Relative energy kick vs distance")
plt.tight_layout()

# 3) Histogram of neighbor‐counts
plt.figure()
plt.hist(stars["n_ngb"], bins=30)
plt.xlabel("Neighbors per star")
plt.ylabel("Events")
plt.title("Neighbor count distribution")
plt.tight_layout()

# 4) Histogram of h_star
plt.figure()
plt.hist(stars["h_star"], bins=30)
plt.xlabel("h [kpc]")
plt.ylabel("Events")
plt.title("Feedback radii distribution")
plt.tight_layout()

# 5) Energy ratio
plt.figure()
plt.hist(stars["E_ratio"], bins=30)
plt.xlabel("E_applied / E_input")
plt.ylabel("Stars")
plt.title("Per‐star energy‐conservation ratio")
plt.tight_layout()

plt.show()
