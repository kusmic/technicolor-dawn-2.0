import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os

# Parameters
snapshot_dir = "output"
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



        # Constants
        gamma = 5.0 / 3.0
        mu = 0.6             # Mean molecular weight (approx. for ionized gas)
        m_H = 1.6726e-24     # Hydrogen mass [g]
        k_B = 1.3807e-16     # Boltzmann constant [erg/K]

        # Convert internal energy to temperature
        T = (gamma - 1.0) * mu * m_H / k_B * u  # Temperature in K

        # Clip temperature to reasonable range
        T_clipped = np.clip(T, 1e2, 1e8)
        log_T = np.log10(T_clipped)
        log_rho = np.log10(rho)

        # Histogram of Internal Energy
        plt.figure(figsize=(8, 5))
        plt.hist(np.log10(u), bins=100, color='gray', alpha=0.7)
        plt.xlabel('log₁₀(InternalEnergy)')
        plt.ylabel('Number of Particles')
        plt.title('Distribution of Internal Energy')
        plt.tight_layout()
        plt.savefig('internal_energy_histogram.png')
        plt.close()

        # Debug: check for extreme internal energy values
        high_energy_mask = u > 1e5
        n_outliers = np.sum(high_energy_mask)
        print(f"[DEBUG] Number of particles with InternalEnergy > 1e5: {n_outliers}")
        if n_outliers > 0:
            print("[DEBUG] Sample of high-energy particles (ρ, T, u):")
            for i in np.where(high_energy_mask)[0][:10]:
                print(f"  rho = {rho[i]:.3e}, T = {T[i]:.3e}, u = {u[i]:.3e}")

        # Plot ρ–T phase diagram
        plt.figure(figsize=(7, 6))
        plt.scatter(log_rho, log_T, s=1, alpha=0.4)
        plt.xlabel(r'$\log_{10}(\rho)$ [g/cm$^3$]')
        plt.ylabel(r'$\log_{10}(T)$ [K]')
        plt.xlim(-8, 2)
        plt.ylim(2, 8)
        plt.title('Density–Temperature Phase Space')
        plt.tight_layout()
        plt.savefig('rho_T_phase.png')
        plt.close()
