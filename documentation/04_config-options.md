# Feedback Configuration Options (Extended Documentation)

This section documents configuration options that appear in `Template-Config.sh` but were previously undocumented.

## `ALLOCATE_SHARED_MEMORY_VIA_POSIX`
- **Description**: Enables shared memory allocation using POSIX-compliant `mmap` rather than `shmget`.
- **Use Case**: Useful when `shmget` is restricted or behaves unpredictably on a given system.
- **Default**: Disabled.

## `LIGHTCONE_MULTIPLE_ORIGINS`
- **Description**: Allows the simulation to use multiple origins when building lightcones.
- **Use Case**: Useful for statistical studies needing multiple mock observer positions.
- **Default**: Disabled.

## `LIGHTCONE_PARTICLES_SKIP_SAVING`
- **Description**: Skips saving lightcone particles to disk.
- **Use Case**: Speeds up runtime when disk I/O is a bottleneck and lightcone particles are not needed for analysis.
- **Default**: Disabled.

## `MPI_HYPERCUBE_ALLTOALL`
- **Description**: Enables a custom hypercube-based implementation of the `MPI_Alltoall` communication pattern.
- **Use Case**: May improve scalability or memory efficiency on systems where default MPI all-to-all performs poorly.
- **Default**: Disabled.

## `OLDSTYLE_SHARED_MEMORY_ALLOCATION`
- **Description**: Uses the original shared memory allocation scheme from early Gadget versions.
- **Use Case**: Maintains compatibility with older system behavior or debug reproducibility.
- **Default**: Disabled.

## `MPI_HYPERCUBE_ALLTOALL`
- **Description**: Enables a custom hypercube-based implementation of the MPI_Alltoall communication pattern.
- **Use Case**: May improve scalability or memory efficiency on systems where default MPI all-to-all performs poorly.
- **Default**: Disabled.

---

Code Configuration
==================

[TOC]

Many aspects of GADGET-4 are controlled with compile-time options [...original content continues here unchanged...]

