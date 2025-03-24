import numpy as np

# Input and output file paths
input_file = "COM_PowerSpect_CMB-TT-full_R3.01.txt"
output_file = "Planck_PR3_COM_PowerSpect_CMB-TT-full_R3.01.txt"

# Load the data, skipping comment lines
data = np.loadtxt(input_file, comments="#")

# Extract the first two columns: ell (multipole) and C_ell
ell = data[:, 0]  # First column: multipole moment ℓ
C_ell = data[:, 1]  # Second column: power spectrum C_ℓ

# Convert C_ell to dimensionless power spectrum if needed
# Gadget-4 typically expects P(k), but we only process C_ell here
# Additional conversions may be needed in N-GenIC or other IC tools

# Save in Gadget-4 compatible format
np.savetxt(output_file, np.column_stack([ell, C_ell]), fmt="%.8e", header="ell  C_ell", comments="")

print(f"✅ Processed power spectrum saved to: {output_file}")
