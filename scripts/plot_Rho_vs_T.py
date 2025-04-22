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

# Gather snapshot files
snapshots = sorted(glob.glob("../output/snapshot_*.hdf5"))

fig, ax = plt.subplots(figsize=(7, 6))
sc = None

# Preload data for animation
snapshot_data = []
for snapfile in snapshots:
    with h5py.File(snapfile, "r") as snap:
        header = dict(snap["Header"].attrs)
        a = header["Time"]
        
        # Unit conversions from header
        UnitMass = header["UnitMass_in_g"]         # grams per code mass unit
        UnitLen = header["UnitLength_in_cm"]       # cm per code length unit
        UnitVel = header["UnitVelocity_in_cm_per_s"]  # cm/s per code velocity unit
        
        # Load data: convert to cgs
        rho_code = np.array(snap["PartType0/Density"]) * a**3
        rho_cgs = rho_code * (UnitMass / UnitLen**3)
        
        u_code = np.array(snap["PartType0/InternalEnergy"])
        u_cgs = u_code * UnitVel**2  # erg/g
        
        # Compute temperature
        T = u_cgs * (gamma - 1.0) * mu * m_H / k_B
        
        # Debug prints
        print(f"[INFO] Snapshot {snapfile}:")
        print(f"  rho_cgs min/max: {rho_cgs.min():.2e} / {rho_cgs.max():.2e} g/cm³")
        print(f"  u_cgs min/max:  {u_cgs.min():.2e} / {u_cgs.max():.2e} erg/g")
        print(f"  T min/max:      {T.min():.2e} / {T.max():.2e} K")
        
        snapshot_data.append((rho_cgs, T, os.path.basename(snapfile)))

def update(frame):
    rho, T, snapname = snapshot_data[frame]
    ax.clear()
    ax.loglog(rho, T, 'o', markersize=2, alpha=0.7)
    ax.set_xlabel("Density [g/cm³]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(f"ρ–T diagram: {snapname}")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs("rho_T_frames", exist_ok=True)
    plt.savefig(f"rho_T_frames/{snapname}.png", dpi=150)

# Run animation
ani = animation.FuncAnimation(fig, update, frames=len(snapshot_data), interval=500)

plt.show()

# Save animation as MP4
ani.save("rho_T_animation.mp4", writer="ffmpeg", dpi=200)
print("Animation saved as rho_T_animation.mp4")
