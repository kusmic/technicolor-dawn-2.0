import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.animation as animation

# Constants
k_B = 1.380649e-16  # Boltzmann constant in erg/K
m_H = 1.6735575e-24  # mass of hydrogen atom in g
mu = 0.6              # mean molecular weight for fully ionized gas
gamma = 5.0 / 3.0      # adiabatic index for monoatomic gas

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

    rho = np.array(snap["PartType0/Density"]) * a**3
    u = np.array(snap["PartType0/InternalEnergy"])
    u_cgs = u * UnitEnergy_in_cgs
    T = (gamma - 1.0) * mu * m_H / k_B * u_cgs

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

    bad_temp_mask = (T < 1) | (T > 1e10)
    if np.any(bad_temp_mask):
        print(f"[WARNING] {np.sum(bad_temp_mask)} particles have unphysical temperatures.")

    snapshot_data.append((rho, T, snapnum))
    snap.close()

def update(frame):
    global sc
    rho, T, snapnum = snapshot_data[frame]
    ax.clear()
    ax.loglog(rho, T, ',', alpha=0.5)
    ax.set_xlabel("Density [g/cm³]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(f"\u03c1–T diagram from snapshot {snapnum}")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"rho_T_frames/frame_{snapnum:04d}.png", dpi=150)

ani = animation.FuncAnimation(fig, update, frames=len(snapshot_data), interval=500)

plt.show()

# Save the animation as MP4 (requires ffmpeg)
ani.save("rho_T_animation.mp4", writer="ffmpeg", dpi=200)
print("Animation saved as rho_T_animation.mp4")
