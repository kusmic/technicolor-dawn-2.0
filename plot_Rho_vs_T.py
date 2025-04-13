import h5py
import numpy as np
import matplotlib.pyplot as plt

# Constants
k_B = 1.380649e-16  # Boltzmann constant in erg/K
m_H = 1.6735575e-24  # mass of hydrogen atom in g
mu = 0.6              # mean molecular weight for fully ionized gas
gamma = 5.0 / 3.0      # adiabatic index for monoatomic gas

# Placeholder: update to match your simulation's energy unit
UnitEnergy_in_cgs = 1.0  # 1.0 means u is already in erg/g

# Load snapshot
snap = h5py.File("output/snapshot_045.hdf5", "r")
header = dict(snap["Header"].attrs)

scale_factor = header["Time"]
a = scale_factor

# Load data
rho = np.array(snap["PartType0/Density"]) * a**3
u = np.array(snap["PartType0/InternalEnergy"])  # in code units or erg/g

# Convert internal energy to CGS if necessary
u_cgs = u * UnitEnergy_in_cgs

# Compute temperature
T = (gamma - 1.0) * mu * m_H / k_B * u_cgs

# Clip extreme internal energies to avoid crazy temperatures
clip_threshold = 1e5
num_high = np.sum(u > clip_threshold)
if num_high > 0:
    print(f"[DEBUG] Number of particles with InternalEnergy > {clip_threshold:.0e}: {num_high}")
    sample = np.where(u > clip_threshold)[0][:10]
    for i in sample:
        print(f"  rho = {rho[i]:.3e}, T = {T[i]:.3e}, u = {u[i]:.3e}")
    u = np.clip(u, None, clip_threshold)
    u_cgs = u * UnitEnergy_in_cgs
    T = (gamma - 1.0) * mu * m_H / k_B * u_cgs

# Check for unphysical temperatures
bad_temp_mask = (T < 1) | (T > 1e10)
if np.any(bad_temp_mask):
    print(f"[WARNING] {np.sum(bad_temp_mask)} particles have unphysical temperatures.")

# Make the ρ–T diagram
plt.figure(figsize=(7, 6))
plt.loglog(rho, T, ',', alpha=0.5)
plt.xlabel("Density [g/cm³]")
plt.ylabel("Temperature [K]")
plt.title("ρ–T diagram from Gadget snapshot")
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("rho_T_diagram.png", dpi=200)
plt.show()

snap.close()
