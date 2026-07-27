# Figure Ideas and the story that they tell 

Ok, so for the lightcone paper, we want to figure out the story with figures. The point of the paper is to 

1.  release a set of convergence, tau and tsz maps at 256 variations in 30 astrophysical parameter dimensions. 
2. Compute the effect of astrophysics on various weak lensing statistics, tau statistics and ksz statistics, and cross statistics. 
3. To try to understand the latent space of the various statistics. 
4. To release a TNG emulator for the various statistics that we look at. 

Ok, so given this, the sections should be. 

1. Methods
    a. BIND recap
    b. TNG simulations and conditioning generation
    c. Generated halos across the sobol sequence
    d. Ray tracing and map generation
2. Validation
    a. Validate various relations? Y-M, f_b, etc? 
    b. validate all maps against IllustrisTNG ray traced maps at various levels of statistics (Cl, peaks, MF, wavelet scattering, f_b? other maps that we can think of. ) -- TODO: compute these maps for the full TNG. As of now, TNG is really replaced with all 10^13 and above halos, so we should see how far off this is. 
3. Astrophysical effects
    a. correlation of WL statistics with respect to the parameters in the sobol sequence? 
    b. Dominant explanations of the sources of dominant parameters, and comparisons between different statistics. 
    c. PCA and latent space construction for the various statistics
4. Emulating the statistics
    a. Building a simple GP emulator
    b. Validation of emulator
    c. example constraint on astro parameters with the emulator
5. Caveats 
6. Future outlook
7. Conclusion
