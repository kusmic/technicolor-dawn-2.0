import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.animation as animation

# Constants
snapshots = sorted(glob.glob("output/snapshot_*.hdf5"))
sfrd_values = []
redshifts = []

# Load data
for snapfile in snapshots:
    with h5py.File(snapfile, "r") as snap:
        header = dict(snap["Header"].attrs)
        a = header["Time"]
        redshift = 1.0 / a - 1.0
        boxsize_kpc = header["BoxSize"]  # comoving, in code units
        boxsize_Mpc = boxsize_kpc / 1e3  # convert to Mpc/h
        volume_Mpc3 = boxsize_Mpc**3     # comoving volume in (Mpc/h)^3

        try:
            sfr = np.array(snap["PartType0/StarFormationRate"])
            total_sfr = np.sum(sfr)  # in Msun/yr
        except KeyError:
            total_sfr = 0.0

        sfrd = total_sfr / volume_Mpc3  # Msun/yr/(Mpc/h)^3
        sfrd_values.append(sfrd)
        redshifts.append(redshift)

# Sort by decreasing redshift
redshifts, sfrd_values = zip(*sorted(zip(redshifts, sfrd_values), reverse=True))
redshifts = np.array(redshifts)
sfrd_values = np.array(sfrd_values)

# Static plot
plt.figure(figsize=(8,6))
plt.plot(redshifts, sfrd_values, marker='o')
plt.xlabel("Redshift (z)")
plt.ylabel("SFR Density [Msun/yr/(Mpc/h)^3]")
plt.yscale("log")
plt.gca().invert_xaxis()
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.title("Star Formation Rate Density vs Redshift")
plt.tight_layout()
plt.savefig("sfrd_vs_redshift.png", dpi=200)
plt.show()

# Animate the growth of the line
fig, ax = plt.subplots(figsize=(8,6))
line, = ax.plot([], [], 'o-', lw=2)
ax.set_xlabel("Redshift (z)")
ax.set_ylabel("SFR Density [Msun/yr/(Mpc/h)^3]")
ax.set_yscale("log")
ax.invert_xaxis()
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.set_xlim(redshifts[0], redshifts[-1])
ax.set_ylim(max(min(sfrd_values[sfrd_values > 0]) * 0.1, 1e-6), max(sfrd_values)*1.2)
ax.set_title("Star Formation History (Build-up)")

def update(i):
    line.set_data(redshifts[:i+1], sfrd_values[:i+1])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(redshifts), interval=500, blit=True)

# Save as mp4
os.makedirs("animations", exist_ok=True)
ani.save("animations/sfrd_vs_redshift.mp4", writer="ffmpeg", dpi=200)
print("✅ Saved: sfrd_vs_redshift.png and sfrd_vs_redshift.mp4")
