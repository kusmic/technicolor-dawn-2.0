import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
import glob

def create_temperature_map(snapshot_file, slice_thickness=0.1, resolution=500):
    """Create a temperature map from a snapshot file"""
    with h5py.File(snapshot_file, 'r') as f:
        # Extract header info (for title)
        time = f['Header'].attrs['Time']  # Scale factor or time
        
        # Extract gas particle data
        pos = f['PartType0/Coordinates'][:]  # Gas particle positions
        mass = f['PartType0/Masses'][:]      # Gas particle masses
        
        # If temperature is directly available
        if 'PartType0/Temperature' in f:
            temp = f['PartType0/Temperature'][:]
        else:
            # Calculate temperature from internal energy if needed
            u = f['PartType0/InternalEnergy'][:]
            # You'll need to convert u to T based on your simulation's EOS
            # This is a placeholder conversion:
            temp = u * 10000  # Adjust this conversion as needed
        
        # Create 2D histogram (projection)
        # Select particles in a thin slice around z=0
        mask = np.abs(pos[:, 2]) < slice_thickness
        
        # Create temperature map
        x_edges = np.linspace(-20, 20, resolution)  # Adjust range as needed
        y_edges = np.linspace(-20, 20, resolution)  # Adjust range as needed
        
        H, _, _ = np.histogram2d(
            pos[mask, 0], pos[mask, 1], 
            bins=[x_edges, y_edges], 
            weights=temp[mask]*mass[mask]
        )
        
        # Mass map for normalization
        M, _, _ = np.histogram2d(
            pos[mask, 0], pos[mask, 1], 
            bins=[x_edges, y_edges], 
            weights=mass[mask]
        )
        
        # Mass-weighted temperature
        temp_map = np.zeros_like(H)
        nonzero = M > 0
        temp_map[nonzero] = H[nonzero] / M[nonzero]
        
        return temp_map, x_edges, y_edges, time

# Get all snapshot files, sorted by number
snapshot_files = sorted(glob.glob("output/snapshot_*.hdf5"))
print(f"Found {len(snapshot_files)} snapshot files")

# Temperature range for consistent colormap
vmin, vmax = 1e4, 1e7

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Initialize the plot with the first frame
temp_map, x_edges, y_edges, time = create_temperature_map(snapshot_files[0])
mesh = ax.pcolormesh(x_edges, y_edges, temp_map.T, 
                    norm=LogNorm(vmin=vmin, vmax=vmax), 
                    cmap='inferno')
cbar = fig.colorbar(mesh, label='Temperature [K]')
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
title = ax.set_title(f'Gas Temperature Map (z=0 slice) - Time: {time:.3f}')
time_text = ax.text(0.05, 0.95, f"Time: {time:.3f}", transform=ax.transAxes,
                  fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

def update(frame):
    filename = snapshot_files[frame]
    print(f"Processing frame {frame+1}/{len(snapshot_files)}: {filename}")
    
    # Create new temperature map
    temp_map, x_edges, y_edges, time = create_temperature_map(filename)
    
    # Update the plot
    mesh.set_array(temp_map.T.ravel())
    title.set_text(f'Gas Temperature Map (z=0 slice) - Time: {time:.3f}')
    time_text.set_text(f"Time: {time:.3f}")
    
    return mesh, title, time_text

# Create the animation
print("Creating animation...")
ani = animation.FuncAnimation(fig, update, frames=len(snapshot_files), 
                              blit=True, interval=200)

# Save the animation
print("Saving animation to temperature_evolution.mp4...")
ani.save("temperature_evolution.mp4", writer='ffmpeg', dpi=150)

# Show the animation (optional - you might want to comment this out for large animations)
plt.show()

print("Done!")