import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
import glob
import os

def create_temperature_map(snapshot_file, slice_thickness=0.5, resolution=200):  # Increased slice thickness, reduced resolution
    """Create a temperature map from a snapshot file"""
    print(f"\nProcessing {snapshot_file}...")
    
    try:
        with h5py.File(snapshot_file, 'r') as f:
            # Debug info omitted for brevity
            
            # Extract header info
            time = f['Header'].attrs['Time']
            
            # Extract gas particle data
            pos = f['PartType0/Coordinates'][:]
            mass = f['PartType0/Masses'][:]
            
            # Get temperature data
            if 'PartType0/Temperature' in f:
                temp = f['PartType0/Temperature'][:]
            else:
                if 'PartType0/InternalEnergy' in f:
                    u = f['PartType0/InternalEnergy'][:]
                    temp = u * 10000  # Adjust conversion factor as needed
                else:
                    return np.zeros((resolution, resolution)), np.linspace(-20, 20, resolution), np.linspace(-20, 20, resolution), time
            
            # Create 2D histogram (projection) - use thicker slice
            mask = np.abs(pos[:, 2]) < slice_thickness
            print(f"Particles in slice: {np.sum(mask)} out of {len(mask)} ({np.sum(mask)/len(mask)*100:.2f}%)")
            
            if np.sum(mask) == 0:
                return np.zeros((resolution, resolution)), np.linspace(-20, 20, resolution), np.linspace(-20, 20, resolution), time
            
            # Determine ranges based on particle positions
            x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
            y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
            
            # Add buffer
            x_buffer = (x_max - x_min) * 0.05
            y_buffer = (y_max - y_min) * 0.05
            
            x_edges = np.linspace(x_min - x_buffer, x_max + x_buffer, resolution)
            y_edges = np.linspace(y_min - y_buffer, y_max + y_buffer, resolution)
            
            # Create temperature map
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
            
            # Mass-weighted temperature, with Gaussian smoothing
            temp_map = np.zeros_like(H)
            nonzero = M > 0
            temp_map[nonzero] = H[nonzero] / M[nonzero]
            
            # Apply Gaussian smoothing to make the map more continuous
            from scipy.ndimage import gaussian_filter
            temp_map = gaussian_filter(temp_map, sigma=1.0)
            
            # Print some diagnostics
            if np.any(temp_map > 0):
                print(f"Temperature range: [{temp_map[temp_map > 0].min():.2e}, {temp_map.max():.2e}]")
            
            return temp_map, x_edges, y_edges, time
            
    except Exception as e:
        print(f"ERROR processing {snapshot_file}: {str(e)}")
        return np.zeros((resolution, resolution)), np.linspace(-20, 20, resolution), np.linspace(-20, 20, resolution), 0

# Get snapshot files
snapshot_pattern = "snapshot_*.hdf5"
snapshot_files = sorted(glob.glob(snapshot_pattern))
print(f"Found {len(snapshot_files)} snapshot files matching pattern: {snapshot_pattern}")

if len(snapshot_files) == 0:
    print("No snapshot files found! Please check the file pattern and directory.")
    exit(1)

# Process first snapshot
temp_map, x_edges, y_edges, time = create_temperature_map(snapshot_files[0])

# Determine temperature range
if np.any(temp_map > 0):
    vmin = temp_map[temp_map > 0].min()
    vmax = temp_map.max()
else:
    vmin, vmax = 1e4, 1e7

# Set up the figure and axis - use a larger figure size
fig, ax = plt.subplots(figsize=(12, 10))

# Use imshow instead of pcolormesh for a more continuous representation
# Transpose and flip the data to match the correct orientation
extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
im = ax.imshow(temp_map.T, origin='lower', extent=extent,
               norm=LogNorm(vmin=vmin, vmax=vmax), 
               cmap='nipy_spectral', interpolation='gaussian', aspect='auto')

# Use a colorbar with a better format
cbar = fig.colorbar(im, label='Temperature [K]', format='%.1e')
ax.set_xlabel('x [kpc]', fontsize=14)
ax.set_ylabel('y [kpc]', fontsize=14)
title = ax.set_title(f'Gas Temperature Map (z=0 slice) - Time: {time:.3f}', fontsize=16)
time_text = ax.text(0.05, 0.95, f"Time: {time:.3f}", transform=ax.transAxes,
                  fontsize=14, bbox=dict(facecolor='white', alpha=0.7))

# Add grid for better spatial reference
ax.grid(True, color='white', alpha=0.3, linestyle=':')

def update(frame):
    filename = snapshot_files[frame]
    print(f"\nProcessing frame {frame+1}/{len(snapshot_files)}: {filename}")
    
    # Create new temperature map
    temp_map, x_edges, y_edges, time = create_temperature_map(filename)
    
    # Clear the axis
    ax.clear()
    
    # Create a new image with the updated data
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax.imshow(temp_map.T, origin='lower', extent=extent,
                  norm=LogNorm(vmin=vmin, vmax=vmax), 
                  cmap='nipy_spectral', interpolation='gaussian', aspect='auto')
    
    # Add grid
    ax.grid(True, color='white', alpha=0.3, linestyle=':')
    
    # Update labels and title
    ax.set_xlabel('x [kpc]', fontsize=14)
    ax.set_ylabel('y [kpc]', fontsize=14)
    title = ax.set_title(f'Gas Temperature Map (z=0 slice) - Time: {time:.3f}', fontsize=16)
    time_text = ax.text(0.05, 0.95, f"Time: {time:.3f}", transform=ax.transAxes,
                      fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
    
    return [im, title, time_text]

# Create output directory
os.makedirs("temp_diagnostics", exist_ok=True)

# Save diagnostic frames - using improved visualization
print("\nSaving diagnostic frames...")
for i, filename in enumerate(snapshot_files):
    temp_map, x_edges, y_edges, time = create_temperature_map(filename)
    
    plt.figure(figsize=(12, 10))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    plt.imshow(temp_map.T, origin='lower', extent=extent,
               norm=LogNorm(vmin=vmin, vmax=vmax), 
               cmap='nipy_spectral', interpolation='gaussian', aspect='auto')
    plt.colorbar(label='Temperature [K]', format='%.1e')
    plt.xlabel('x [kpc]', fontsize=14)
    plt.ylabel('y [kpc]', fontsize=14)
    plt.title(f'Gas Temperature Map (z=0 slice) - Time: {time:.3f}', fontsize=16)
    plt.grid(True, color='white', alpha=0.3, linestyle=':')
    
    plt.savefig(f"temp_diagnostics/frame_{i:04d}.png", dpi=150)
    plt.close()

# Create animation
print("\nCreating animation...")
ani = animation.FuncAnimation(fig, update, frames=len(snapshot_files), 
                              blit=False, interval=200)

# Save the animation
print("Saving animation to temperature_evolution.mp4...")
try:
    ani.save("temperature_evolution.mp4", writer='ffmpeg', dpi=150)
    print("Animation saved successfully!")
except Exception as e:
    print(f"Error saving animation: {str(e)}")
    print("Check the individual frames in the temp_diagnostics folder.")

plt.show()
print("Done!")