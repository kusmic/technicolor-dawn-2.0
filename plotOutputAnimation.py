import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define colors (and optional marker sizes) for each PartType.
# Modify these if your simulation uses different types or you wish different styling.
part_colors = {
    'PartType0': {'name': 'Gas', 'color': 'blue', 'size': 2},
    'PartType1': {'name': 'Dark Matter', 'color': 'black', 'size': 2},
    'PartType2': {'name': 'Disk', 'color': 'brown', 'size': 3},
    'PartType3': {'name': 'Bulge', 'color': 'orange', 'size': 3},
    'PartType4': {'name': 'Stars', 'color': 'red', 'size': 5},
    'PartType5': {'name': 'Black Holes', 'color': 'purple', 'size': 3},
    'PartType6': {'name': 'Dust', 'color': 'green', 'size': 5}
}

# Change the glob pattern to match the snapshot filenames
snapshot_files = sorted(glob.glob("output/snap*.hdf5"))

# Set up the figure and axis.
fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    ax.clear()
    filename = snapshot_files[frame]
    with h5py.File(filename, 'r') as f:
        # Loop through our defined particle types.
        for part, props in part_colors.items():
            if part in f:
                # Read the particle coordinates.
                coords = f[part]['Coordinates'][:]  # Expecting shape (N, 3)
                # Plot only x and y (2D projection).
                ax.scatter(coords[:, 0], coords[:, 1],
                           s=props['size'],
                           c=props['color'],
                           label=props['name'],
                           alpha=0.7)
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title(f"Snapshot {frame}: {filename}")
    # Optionally, set the axis limits based on your simulation scale.
    # ax.set_xlim(-500, 500)
    # ax.set_ylim(-500, 500)
    ax.legend(loc='upper right')

# Create the animation.
ani = animation.FuncAnimation(fig, update, frames=len(snapshot_files), interval=500)

plt.show()

# Optionally, to save the animation as a video (requires ffmpeg):
ani.save("gadget4_animation.mp4", writer="ffmpeg", dpi=200)