import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

# Parameters
snapshot_dir = "./output"
snapshots = sorted(glob.glob(f"{snapshot_dir}/snap*.hdf5"))
num_bins = 200
density_floor = 1e-6
energy_floor = 1e-12
save_anim = True  # Set to False to skip saving the animation

# Preload snapshot data
all_data = []

for snap in snapshots:
    with h5py.File(snap, 'r') as f:

        time = f['Header'].attrs['Time']  # Scale factor
        redshift = 1.0/time - 1.0

        rho = f['PartType0']['Density'][:] * f['Header'].attrs['Time']**-3
        try:
            u = f['PartType0']['InternalEnergy'][:]
        except KeyError:
            entropy = f['PartType0']['Entropy'][:]
            rho_phys = rho
            u = entropy * (rho_phys)**(2/3) / (5./3 - 1)  # assuming gamma = 5/3

        temp = (u * (5./3 - 1))  # assuming μ, k_B, etc. folded into units
        mask = (rho > density_floor) & (temp > energy_floor)
        all_data.append((rho[mask], temp[mask]))

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.hexbin(all_data[0][0], all_data[0][1], gridsize=num_bins, 
               bins='log', cmap='plasma', mincnt=1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Density [code units]")
ax.set_ylabel("Temperature [code units]")
cb = fig.colorbar(sc, ax=ax, label="log(N)")

# Animation update function
def update(i):
    ax.clear()
    rho, temp = all_data[i]
    sc = ax.hexbin(rho, temp, gridsize=num_bins, bins='log', cmap='plasma', mincnt=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Density [code units]")
    ax.set_ylabel("Temperature [code units]")
    ax.set_title(f"Snapshot {i}, z={redshift:.2f}")
    return sc,

# Create animation
print("\nCreating animation...")
ani = animation.FuncAnimation(fig, update, frames=len(all_data), interval=300)

# Save or show
if save_anim:
    ani.save("rho_T_evolution.mp4", writer="ffmpeg", dpi=150)
else:
    plt.show()