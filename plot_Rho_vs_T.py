import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio

# Constants
k_B = 1.380649e-16  # Boltzmann constant in erg/K
m_H = 1.6735575e-24  # mass of hydrogen atom in g
mu = 0.6              # mean molecular weight for fully ionized gas
gamma = 5.0 / 3.0      # adiabatic index for monoatomic gas

# Placeholder: update to match your simulation's energy unit
UnitEnergy_in_cgs = 1.0  # 1.0 means u is already in erg/g

# Gather snapshot files
snapshots = sorted(glob.glob("output/snapshot_*.hdf5"))
output_dir = "rho_T_frames"
os.makedirs(output_dir, exist_ok=True)

for snapfile in snapshots:
    snap = h5py.File(snapfile, "r")
    header = dict(snap["Header"].attrs)

    scale_factor = header["Time"]
    a = scale_factor
    snapnum = snapfile.split("_")[-1].split(".")[0]

    # Load data
    rho = np.array(snap["PartType0/Density"]) * a**3
    u = np.array(snap["PartType0/InternalEnergy"])  # in code units or erg/g

    # Convert internal energy to CGS if necessary
    u_cgs = u * UnitEnergy_in_cgs

    # Compute temperature
    T = (gamma - 1.0) * mu * m_H / k_B * u_cgs

    # Clip extreme internal energies to avoid crazy temperatures
    clip_threshold = 1e5
    num_high = np.sum(u > clip_threshold)
    print(f"[DEBUG] Number of particles with InternalEnergy > {clip_threshold:.0e}: {num_high}")
    if num_high > 0:
        sample = np.where(u > clip_threshold)[0][:10]
        for i in sample:
            print(f"  rho = {rho[i]:.3e}, T = {T[i]:.3e}, u = {u[i]:.3e}")
        u = np.clip(u, None, clip_threshold)
        u_cgs = u * UnitEnergy_in_cgs
        T = (gamma - 1.0) * mu * m_H / k_B * u_cgs

    # Check for unphysical temperatures
    bad_temp_mask = (T < 1) | (T > 1e10)
    if np.any(bad_temp_mask):
        print(f"[WARNING] {np.sum(bad_temp_mask)} particles have unphysical temperatures.")

    # Make the ρ–T diagram
    plt.figure(figsize=(7, 6))
    plt.loglog(rho, T, ',', alpha=0.5)
    plt.xlabel("Density [g/cm³]")
    plt.ylabel("Temperature [K]")
    plt.title(f"ρ–T diagram from snapshot {snapnum}")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    frame_filename = os.path.join(output_dir, f"frame_{snapnum}.png")
    plt.savefig(frame_filename, dpi=200)
    plt.close()
    snap.close()

# Create MP4 animation
frames = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
images = [imageio.imread(frame) for frame in frames]
imageio.mimsave("rho_T_animation.mp4", images, fps=4)
print("Animation saved as rho_T_animation.mp4")
