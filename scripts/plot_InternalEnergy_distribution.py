import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot gas internal energy distribution for a given snapshot")
    parser.add_argument("snapshot", type=int, help="Snapshot number (e.g., 47 for snapshot_047.hdf5)")
    parser.add_argument("cutoff", type=float, help="Clamp cutoff value to mark on the distribution")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins (default: 100)")
    parser.add_argument("--min_exp", type=float, default=-2, help="Minimum exponent for log-spaced bins (default: -2)")
    parser.add_argument("--max_exp", type=float, default=6, help="Maximum exponent for log-spaced bins (default: 6)")
    parser.add_argument("--path", type=str, default="output", help="Directory containing snapshot files")
    args = parser.parse_args()

    filename = f"{args.path}/snapshot_{args.snapshot:03d}.hdf5"
    with h5py.File(filename, "r") as f:
        u = f["PartType0"]["InternalEnergy"][:]

    bins = np.logspace(args.min_exp, args.max_exp, args.bins)
    plt.figure(figsize=(8, 6))
    plt.hist(u, bins=bins)
    plt.xscale("log")
    plt.xlabel("Internal Energy (u)")
    plt.ylabel("Number of Gas Particles")
    plt.title(f"Gas Internal Energy Distribution: Snapshot {args.snapshot:03d}")
    plt.axvline(args.cutoff, color="r", linestyle="--", label=f"Clamp = {args.cutoff:.2e}")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
