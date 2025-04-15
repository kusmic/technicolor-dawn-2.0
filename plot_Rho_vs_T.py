import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.animation as animation

# Constants
k_B = 1.380649e-16     # Boltzmann constant in erg/K
m_H = 1.6735575e-24    # Mass of hydrogen atom in g
mu = 0.6               # Mean molecular weight for fully ionized gas
gamma = 5.0 / 3.0      # Adiabatic index for monoatomic gas

# Simulation unit for internal energy
UnitEnergy_in_cgs = 1.989e53  # erg/g

# Gather snapshot files
snapshots = sorted(glob.glob("output/snapshot_*.hdf5"))

fig, ax = plt.subplots(figsize=(7, 6))
sc = None

# Preload data for animation
snapshot_data = []
for snapfile in snapshots:
    snap = h5py.File(snapfile, "r")
    header = dict(snap["Header"].attrs)
    a = header["Time"]
    snapnum = snapfile.split("_")[-1].split(".")[0]

    # Load data
    rho = np.array(snap["PartType0/Density"]) * a**3
    u = np.array(snap["PartType0/InternalEnergy"])

    # Clip extreme internal energies BEFORE computing T
    clip_threshold = 1e5
    num_high = np.sum(u > clip_threshold)
    if num_high > 0:
        print(f"[DEBUG] Number of particles with InternalEnergy > {clip_threshold:.0e}: {num_high}")
        sample = np.where(u > clip_threshold)[0][:10]
        for i in sample:
            print(f"  rho = {rho[i]:.3e}, u = {u[i]:.3e} (before clip)")
        u = np.clip(u, None, clip_threshold)

    # Convert internal energy and compute temperature
    u_cgs = u * UnitEnergy_in_cgs
    T = (gamma - 1.0) * mu * m_H / k_B * u_cgs

    # Sanity checks
    print(f"[INFO] u min/max: {u.min():.2e} / {u.max():.2e}")
    print(f"[INFO] T min/max: {T.min():.2e} / {T.max():.2e}")

    # Warn on unphysical temperatures
    bad_temp_mask = (T < 1) | (T > 1e10)
    if np.any(bad_temp_mask):
        print(f"[WARNING] {np.sum(bad_temp_mask)} particles have unphysical temperatures.")

    snapshot_data.append((rho, T, snapnum))
    snap.close()

def update(frame):
    global sc
    rho, T, snapnum = snapshot_data[frame]
    ax.clear()
    ax.loglog(rho, T, 'o', markersize=2, alpha=0.7)
    ax.set_xlabel("Density [g/cm³]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(f"\u03c1–T diagram from snapshot {snapnum}")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    # Save each frame as PNG
    os.makedirs("rho_T_frames", exist_ok=True)
    plt.savefig(f"rho_T_frames/frame_{snapnum}.png", dpi=150)

# Run animation
ani = animation.FuncAnimation(fig, update, frames=len(snapshot_data), interval=500)

plt.show()

# Save animation as MP4
ani.save("rho_T_animation.mp4", writer="ffmpeg", dpi=200)
print("Animation saved as rho_T_animation.mp4")
