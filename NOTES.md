NOTES

So for grav_direct I need to make sure there is refactored CUDA device code for:

- Sp->nearest_image_intpos_to_pos(...)
Converts integer particle positions to floating-point coordinates, handling periodic boundary conditions.

-get_gfactors_monopole(gfac, r, hmax, rinv)
Computes gravitational force factors for monopole interactions.

-modify_gfactors_pm_monopole(gfac, r, rinv, mfp)
(Conditional, if PMGRID is enabled) Modifies force factors for mesh-based (PM) monopole interactions.

-Ewald.ewald_gridlookup(...)
(Conditional, if DoEwald is true) Computes Ewald summation corrections for periodic boundary conditions.


ALSO: Make a separate .cc (and .h) version of code for CUDA compiling. May need to change to .cu. The files changed thus far:
    - grav_direct
    - gwalk
    - intposconvert


SCIENCE:
    - What is grav. potential in Gadget vs. actual Einstein calc.
    - See environments it messes with
        - dense regions?
        - clustered points?
    - halo spin distribution
    - halo spin correlation (2pcf)