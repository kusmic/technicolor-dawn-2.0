import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("snapshot_040.hdf5", "r") as f:
    u = f["PartType0"]["InternalEnergy"][:]
    plt.hist(u, bins=np.logspace(-2, 6, 100))
    plt.xscale("log")
    plt.title("Gas Internal Energy (u)")
    plt.axvline(1e5, color="r", linestyle="--", label="Clamp")
    plt.legend()
    plt.show()
