import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import os
from tqdm import tqdm
import subprocess

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

def generate_animation():
    # Create output directory
    os.makedirs("temp_frames", exist_ok=True)
    
    # Get all snapshot files, sorted by number
    snapshot_files = sorted(glob.glob("output/snapshot_*.hdf5"))
    
    print(f"Processing {len(snapshot_files)} snapshots...")
    
    # Temperature range for consistent colormap
    vmin, vmax = 1e4, 1e7
    
    # Process each snapshot
    for i, snap_file in enumerate(tqdm(snapshot_files)):
        # Create temperature map
        temp_map, x_edges, y_edges, time = create_temperature_map(snap_file)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x_edges, y_edges, temp_map.T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='inferno')
        cbar = plt.colorbar(label='Temperature [K]')
        plt.xlabel('x [kpc]')
        plt.ylabel('y [kpc]')
        plt.title(f'Gas Temperature Map (z=0 slice) - Time: {time:.3f}')
        
        # Add time counter
        plt.text(0.05, 0.95, f"Time: {time:.3f}", transform=plt.gca().transAxes,
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Save frame
        frame_filename = f"temp_frames/frame_{i:04d}.png"
        plt.savefig(frame_filename, dpi=150)
        plt.close()
    
    print("Converting frames to video...")
    
    # Create video using ffmpeg (ensure ffmpeg is installed)
    try:
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '10',  # Frames per second
            '-i', 'temp_frames/frame_%04d.png',
            '-c:v', 'libx264',
            '-profile:v', 'high',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            'temperature_evolution.mp4'
        ]
        subprocess.run(cmd, check=True)
        print("Video created successfully: temperature_evolution.mp4")
    except Exception as e:
        print(f"Error creating video: {e}")
        print("If ffmpeg is not installed, install it or use an alternative method to create the video.")

if __name__ == "__main__":
    generate_animation()