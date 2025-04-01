import numpy as np
import h5py
import matplotlib.pyplot as plt

# Simulation parameters
box_size = 50  # Box size in kpc/h
num_gas = 1000  # Gas particles
num_dm = 1000  # Dark matter particles
h = 0.678  # Hubble parameter
omega_m = 0.308  # Total matter density (dark matter + baryons)
omega_b = 0.0482  # Baryon density (gas only)

# Generate random positions (assuming uniform distribution)
pos_gas = np.random.uniform(0, box_size, (num_gas, 3))
pos_dm = np.random.uniform(0, box_size, (num_dm, 3))

# Generate random velocities (assuming initial thermal state)
vel_gas = np.random.normal(0, 10, (num_gas, 3))
vel_dm = np.random.normal(0, 10, (num_dm, 3))

# Assign unique particle IDs
particle_ids_gas = np.arange(1, num_gas + 1, dtype=np.uint64)
particle_ids_dm = np.arange(num_gas + 1, num_gas + num_dm + 1, dtype=np.uint64)

rho_crit = 2.775e2 * h**2  # M_sun / (Mpc/h)^3
volume = box_size**3  # Box volume in (Mpc/h)^3

correction_factor = 4.59724025974
mass_gas = ( (rho_crit * omega_b * volume) / num_gas ) / correction_factor
mass_dm = ( (rho_crit * (omega_m - omega_b) * volume) / num_dm ) / correction_factor

print(f"Gas Particle Mass: {mass_gas:.3e} M_sun")
print(f"DM Particle Mass: {mass_dm:.3e} M_sun")
print(f"Total Simulated Omega_m: {(mass_gas*num_gas + mass_dm*num_dm) / (rho_crit * volume)}")
print(f"num_dm * mass_dm: {(mass_dm*num_dm)}")
print(f"mass_gas: {mass_gas}")
print(f"mass_dm: {mass_dm}")
print(f"rho_crit: {rho_crit}")

# Internal energy for gas (thermal state)
internal_energy = np.full(num_gas, 250)  # Example internal energy for gas

# Electron abundance for gas
electron_abundance = np.full(num_gas, 0.1)

# Metallicity for gas
metallicity_gas = np.full(num_gas, 0.02)  # Example metallicity for gas

# Smoothing length for gas
smoothing_length = np.full(num_gas, 0.05)

# Create HDF5 file for Gadget-4
with h5py.File("cosmo_ic.hdf5", "w") as f:
    header = f.create_group("Header")
    header.attrs["NumPart_Total"] = [num_gas, num_dm, 0, 0, 0, 0, 0]
    header.attrs["NumPart_ThisFile"] = [num_gas, num_dm, 0, 0, 0, 0, 0]
    header.attrs["MassTable"] = [0, mass_dm, 0, 0, 0, 0, 0]
    header.attrs["Time"] = 0.0
    header.attrs["Redshift"] = 99.0  # Initial redshift
    header.attrs["BoxSize"] = box_size
    header.attrs["NumFilesPerSnapshot"] = 1  # Added this line

    # PartType0: Gas
    part0 = f.create_group("PartType0")
    part0.create_dataset("Coordinates", data=pos_gas)
    part0.create_dataset("Velocities", data=vel_gas)
    part0.create_dataset("ParticleIDs", data=particle_ids_gas)
    part0.create_dataset("Masses", data=np.full(num_gas, mass_gas))
    part0.create_dataset("InternalEnergy", data=internal_energy)
    part0.create_dataset("Density", data=mass_gas / (smoothing_length ** 3))  # Example density calculation
    part0.create_dataset("SmoothingLength", data=smoothing_length)
    part0.create_dataset("ElectronAbundance", data=electron_abundance)
    part0.create_dataset("Metallicity", data=metallicity_gas)

    # PartType1: Dark Matter
    part1 = f.create_group("PartType1")
    part1.create_dataset("Coordinates", data=pos_dm)
    part1.create_dataset("Velocities", data=vel_dm)
    part1.create_dataset("ParticleIDs", data=particle_ids_dm)
    part1.create_dataset("Masses", data=np.full(num_dm, mass_dm))


plt.scatter(pos_dm[:, 0], pos_dm[:, 1], s=1, alpha=0.5, label="DM")
plt.scatter(pos_gas[:, 0], pos_gas[:, 1], s=1, alpha=0.5, label="Gas")
plt.legend()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Particle Distribution")
plt.show()

print("✅ Gadget-4 IC file 'cosmo_ic.hdf5' created successfully!")