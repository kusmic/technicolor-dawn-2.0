import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define colors (and optional marker sizes) for each PartType.
# Modify these if your simulation uses different types or you wish different styling.
part_colors = {
    'PartType0': {'color': 'blue',   'size': 1},    # Gas
    'PartType1': {'color': 'black',  'size': 1},    # Dark Matter
    'PartType2': {'color': 'green',  'size': 1},    # Disk particles (if any)
    'PartType3': {'color': 'orange', 'size': 1},    # Bulge (if any)
    'PartType4': {'color': 'red',    'size': 1},    # Stars
    'PartType5': {'color': 'purple', 'size': 1},    # Black Holes or other
    'PartType6': {'color': 'brown',  'size': 1}     # Dust
}

# Change the glob pattern to match your snapshot filenames.
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
                           label=part,
                           alpha=0.6)
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title(f"Snapshot {frame}: {filename}")
    # Optionally, set the axis limits based on your simulation scale.
    # ax.set_xlim(-500, 500)
    # ax.set_ylim(-500, 500)
    ax.legend(loc='upper right')

# Create the animation.
ani = animation.FuncAnimation(fig, update, frames=len(snapshot_files), interval=100)

# To display the animation interactively:
plt.show()

# Optionally, to save the animation as a video (requires ffmpeg installed):
# ani.save("gadget4_animation.mp4", writer="ffmpeg", dpi=300)
