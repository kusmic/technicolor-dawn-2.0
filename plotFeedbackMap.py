import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
import glob
import os

def create_temperature_map(snapshot_file, slice_thickness=0.1, resolution=500):
    """Create a temperature map from a snapshot file"""
    print(f"\nProcessing {snapshot_file}...")
    
    try:
        with h5py.File(snapshot_file, 'r') as f:
            # Debug: Check file structure
            print(f"Keys in file: {list(f.keys())}")
            if 'PartType0' not in f:
                print(f"WARNING: No gas particles (PartType0) in file!")
                return np.zeros((resolution, resolution)), np.linspace(-20, 20, resolution), np.linspace(-20, 20, resolution), 0
            
            print(f"Gas particles: {len(f['PartType0/Coordinates'])}")
            print(f"Available fields: {list(f['PartType0'].keys())}")
            
            # Extract header info (for title)
            time = f['Header'].attrs['Time']  # Scale factor or time
            print(f"Snapshot time: {time}")
            
            # Extract gas particle data
            pos = f['PartType0/Coordinates'][:]  # Gas particle positions
            print(f"Position range: x=[{pos[:, 0].min():.2f}, {pos[:, 0].max():.2f}], "
                  f"y=[{pos[:, 1].min():.2f}, {pos[:, 1].max():.2f}], "
                  f"z=[{pos[:, 2].min():.2f}, {pos[:, 2].max():.2f}]")
            
            mass = f['PartType0/Masses'][:]      # Gas particle masses
            print(f"Mass range: [{mass.min():.2e}, {mass.max():.2e}]")
            
            # Get temperature data
            if 'PartType0/Temperature' in f:
                temp = f['PartType0/Temperature'][:]
                print(f"Temperature range: [{temp.min():.2e}, {temp.max():.2e}]")
            else:
                print("No direct temperature field. Using internal energy...")
                if 'PartType0/InternalEnergy' in f:
                    u = f['PartType0/InternalEnergy'][:]
                    print(f"Internal energy range: [{u.min():.2e}, {u.max():.2e}]")
                    # You'll need to convert u to T based on your simulation's EOS
                    # This is a placeholder conversion:
                    temp = u * 10000  # Adjust this conversion as needed
                    print(f"Converted temperature range: [{temp.min():.2e}, {temp.max():.2e}]")
                else:
                    print("ERROR: Neither Temperature nor InternalEnergy found!")
                    return np.zeros((resolution, resolution)), np.linspace(-20, 20, resolution), np.linspace(-20, 20, resolution), time
            
            # Create 2D histogram (projection)
            # Select particles in a thin slice around z=0
            mask = np.abs(pos[:, 2]) < slice_thickness
            print(f"Particles in slice: {np.sum(mask)} out of {len(mask)} ({np.sum(mask)/len(mask)*100:.2f}%)")
            
            if np.sum(mask) == 0:
                print("WARNING: No particles in this slice!")
                return np.zeros((resolution, resolution)), np.linspace(-20, 20, resolution), np.linspace(-20, 20, resolution), time
            
            # Determine sensible grid ranges based on actual particle positions
            x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
            y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
            
            # Add a small buffer
            x_buffer = (x_max - x_min) * 0.05
            y_buffer = (y_max - y_min) * 0.05
            
            x_edges = np.linspace(x_min - x_buffer, x_max + x_buffer, resolution)
            y_edges = np.linspace(y_min - y_buffer, y_max + y_buffer, resolution)
            
            print(f"Grid ranges: x=[{x_edges[0]:.2f}, {x_edges[-1]:.2f}], y=[{y_edges[0]:.2f}, {y_edges[-1]:.2f}]")
            
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
            
            # Mass-weighted temperature
            temp_map = np.zeros_like(H)
            nonzero = M > 0
            temp_map[nonzero] = H[nonzero] / M[nonzero]
            
            print(f"Temperature map statistics:")
            print(f"  Min: {temp_map[temp_map > 0].min() if np.any(temp_map > 0) else 0:.2e}")
            print(f"  Max: {temp_map.max():.2e}")
            print(f"  Mean: {np.mean(temp_map[temp_map > 0]) if np.any(temp_map > 0) else 0:.2e}")
            print(f"  Non-zero cells: {np.count_nonzero(temp_map)} out of {temp_map.size} ({np.count_nonzero(temp_map)/temp_map.size*100:.2f}%)")
            
            return temp_map, x_edges, y_edges, time
            
    except Exception as e:
        print(f"ERROR processing {snapshot_file}: {str(e)}")
        return np.zeros((resolution, resolution)), np.linspace(-20, 20, resolution), np.linspace(-20, 20, resolution), 0

# Get all snapshot files, sorted by number
snapshot_pattern = "snapshot_*.hdf5"  # Adjust this pattern as needed
snapshot_files = sorted(glob.glob("output/"+snapshot_pattern))
print(f"Found {len(snapshot_files)} snapshot files matching pattern: {snapshot_pattern}")

if len(snapshot_files) == 0:
    print("No snapshot files found! Please check the file pattern and directory.")
    exit(1)

# Process the first snapshot to get initial data and determine ranges
temp_map, x_edges, y_edges, time = create_temperature_map(snapshot_files[0])

# Determine temperature range from the first frame
if np.any(temp_map > 0):
    vmin = temp_map[temp_map > 0].min()
    vmax = temp_map.max()
    print(f"Auto-detected temperature range: [{vmin:.2e}, {vmax:.2e}]")
else:
    # Default range if no data
    vmin, vmax = 1e4, 1e7
    print(f"No valid temperature data in first frame. Using default range: [{vmin:.2e}, {vmax:.2e}]")

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Initialize the plot with the first frame
mesh = ax.pcolormesh(x_edges, y_edges, temp_map.T, 
                    norm=LogNorm(vmin=vmin, vmax=vmax), 
                    cmap='rainbow')  # Changed to rainbow colormap
cbar = fig.colorbar(mesh, label='Temperature [K]')
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
title = ax.set_title(f'Gas Temperature Map (z=0 slice) - Time: {time:.3f}')
time_text = ax.text(0.05, 0.95, f"Time: {time:.3f}", transform=ax.transAxes,
                  fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

def update(frame):
    filename = snapshot_files[frame]
    print(f"\nProcessing frame {frame+1}/{len(snapshot_files)}: {filename}")
    
    # Create new temperature map
    temp_map, x_edges, y_edges, time = create_temperature_map(filename)
    
    # Update the plot
    mesh.set_array(temp_map.T.ravel())
    
    # Update x and y ranges if they changed
    mesh.set_extent([x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    
    title.set_text(f'Gas Temperature Map (z=0 slice) - Time: {time:.3f}')
    time_text.set_text(f"Time: {time:.3f}")
    
    print(f"Updated mesh array size: {temp_map.T.size}")
    if np.any(temp_map > 0):
        print(f"Updated temperature range: [{temp_map[temp_map > 0].min():.2e}, {temp_map.max():.2e}]")
    else:
        print("WARNING: Updated temperature map has no positive values!")
    
    return mesh, title, time_text

# Create output directory for diagnostic images (optional)
os.makedirs("temp_diagnostics", exist_ok=True)

# Save individual frames for diagnosis (optional)
print("\nSaving diagnostic frames...")
for i, filename in enumerate(snapshot_files):
    temp_map, x_edges, y_edges, time = create_temperature_map(filename)
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x_edges, y_edges, temp_map.T, 
                  norm=LogNorm(vmin=vmin, vmax=vmax), 
                  cmap='rainbow')
    plt.colorbar(label='Temperature [K]')
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.title(f'Gas Temperature Map (z=0 slice) - Time: {time:.3f}')
    
    plt.savefig(f"temp_diagnostics/frame_{i:04d}.png", dpi=150)
    plt.close()

# Create the animation
print("\nCreating animation...")
ani = animation.FuncAnimation(fig, update, frames=len(snapshot_files), 
                              blit=True, interval=200)

# Save the animation
print("Saving animation to temperature_evolution.mp4...")
try:
    ani.save("temperature_evolution.mp4", writer='ffmpeg', dpi=150)
    print("Animation saved successfully!")
except Exception as e:
    print(f"Error saving animation: {str(e)}")
    print("Try saving individual frames in the temp_diagnostics folder instead.")

# Show the animation (optional - comment out for large animations or headless environments)
print("Displaying animation (close window to continue)...")
plt.show()

print("Done!")