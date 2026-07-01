"""
Gaia DR3 ADQL Query Generator
-----------------------------
Calculates the retro-propagated coordinates and dynamic parallax bounds 
for a given pulsar using PyGaia, and generates the exact ADQL query.

Methodology:
- Past epoch coordinates are derived using EpochPropagation.
- The base distance is constrained by the pulsar's parallax AND its error.
- The physical bounds are expanded by the maximum relative separation velocity (500 km/s).
- An observational buffer is added to absorb Gaia's astrometric uncertainties.

"""

import numpy as np
import pandas as pd
from pygaia.astrometry.coordinates import EpochPropagation

def calculate_past_parameters(row: pd.Series):
    """
    Propagates the pulsar's coordinates to its birth epoch using PyGaia 
    and calculates dynamic parallax limits incorporating the pulsar's intrinsic error.
    """
    ra_rad = np.radians(float(row['ra']))
    dec_rad = np.radians(float(row['dec']))
    plx_now = float(row['parallax'])
    plx_err = float(row['parallax_error'])
    pmra = float(row['pmra'])
    pmdec = float(row['pmdec'])
    rv = float(row['radial_velocity'])
    age_kyr = float(row['age_kyr'])
    
    # Coordinate Propagation to the Past using PyGaia
    age_yr = age_kyr * 1000.0
    past_epoch = 2016.0 - age_yr  # Gaia DR3 reference epoch is J2016.0
    
    ep = EpochPropagation()
    
    ra_past_rad, dec_past_rad = ep.propagate_pos(
        np.array([ra_rad]), 
        np.array([dec_rad]), 
        np.array([plx_now]), 
        np.array([pmra]), 
        np.array([pmdec]), 
        np.array([rv]), 
        2016.0, 
        past_epoch
    )
    
    ra_past = np.degrees(ra_past_rad[0])
    dec_past = np.degrees(dec_past_rad[0])
    
    # Dynamic Parallax Range Calculation (Incorporating Pulsar Error)
    age_myr = age_kyr / 1000.0
    
    # Extremes of the pulsar's possible current parallax
    # We use max() to prevent negative or zero parallaxes in anomalous cases
    plx_pulsar_min = max(plx_now - plx_err, 0.001) 
    plx_pulsar_max = plx_now + plx_err
    
    # Convert pulsar parallax extremes to physical distances (parsecs)
    d_pulsar_max_pc = 1000.0 / plx_pulsar_min
    d_pulsar_min_pc = 1000.0 / plx_pulsar_max
    
    # Maximum assumed relative velocity (350 NS + 150 Companion = 500 km/s)
    v_relative_max_km_s = 500.0
    max_radial_separation_pc = v_relative_max_km_s * age_myr 
    
    # Expand the physical boundaries using the runaway kinematics
    d_cand_max_pc = d_pulsar_max_pc + max_radial_separation_pc
    d_cand_min_pc = max(d_pulsar_min_pc - max_radial_separation_pc, 1.0)
    
    # Convert back to parallax bounds for the candidate
    plx_max_physical = 1000.0 / d_cand_min_pc
    plx_min_physical = 1000.0 / d_cand_max_pc
    
    # Observational Margin (Gaia Error Buffer)
    # Add a generous 1 mas buffer to absorb candidate instrumental errors
    observational_buffer_mas = 1 
    plx_max = plx_max_physical + observational_buffer_mas
    plx_min = plx_min_physical - observational_buffer_mas
    
    return ra_past, dec_past, plx_min, plx_max

def calculate_search_radius(row: pd.Series, v_max_km_s=500.0, buffer=1.5):
    """
    Calculates the dynamic, astrophysically motivated search radius for a cone search.
    
    :param plx_max: Maximum parallax (closest distance limit to Earth) in mas.
    :param age_mean: Estimated age of the remnant in years.
    :param age_error: Uncertainty in the age in years.
    :param v_max_km_s: Conservative upper limit for the velocity of a runaway object.
    :param buffer: Safety factor to absorb astrometric errors (e.g. 20%).
    :return: search_radius_deg (search radius in degrees).
    """
    
    plx_now = float(row['parallax'])
    plx_err = float(row['parallax_error'])
    
    age_mean = float(row['age_kyr'])
    age_error = float(row['age_error_kyr'])

    plx_max = plx_now - plx_err

    # Minimum physical distance to the pulsar (scenario where the object appears fastest)
    d_min_pc = 1000.0 / plx_max
    
    # Maximum possible age (scenario where it had the most time to travel)
    age_max_yr = age_mean + age_error
    
    # Maximum apparent proper motion in arcsec/yr
    # Formula: pm (arcsec/yr) = V_t (km/s) / (4.74 * d (pc))
    pm_max_arcsec_yr = v_max_km_s / (4.74 * d_min_pc)
    
    # Maximum angular distance traveled since the explosion (in degrees)
    max_angular_distance_deg = (pm_max_arcsec_yr * age_max_yr) / 3600.0
    
    # Apply a safety margin (buffer)
    search_radius_deg = max_angular_distance_deg * buffer
    
    return search_radius_deg

def generate_query(target_id: str, csv_file: str = "pulsar_targets.csv", radius_deg: float = 4.0):
    """
    Reads the target catalog, computes the physical constraints, 
    and prints the formatted ADQL query.
    """
    try:
        df_targets = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[ERROR] Target catalog not found: {csv_file}")
        return

    pulsar_data = df_targets[df_targets['Pulsar_ID'] == target_id]
    
    if pulsar_data.empty:
        print(f"[ERROR] Target pulsar '{target_id}' not found in catalog.")
        return
        
    row = pulsar_data.iloc[0]
    
    # Perform kinematic calculations
    ra_past, dec_past, plx_min, plx_max = calculate_past_parameters(row)

    radius_deg = calculate_search_radius(row)
    
    # Display configuration summary
    print("\n" + "="*65)
    print(f" TARGET PULSAR: {target_id} | ESTIMATED AGE: {row['age_kyr']} kyr")
    print("="*65)
    print(f" ► Search Center (Past Epoch): RA = {ra_past:.6f}°, Dec = {dec_past:.6f}°")
    print(f" ► Dynamic Parallax Bounds:    from {plx_min:.6f} to {plx_max:.6f} mas")
    print(f" ► Pulsar Origin Error Margin: Included ({row['parallax_error']} mas intrinsic)")
    print(f" ► Assumed Max Rel. Velocity:  500 km/s (350 NS + 150 Companion)")
    print(f" ► Observational Error Buffer: ± 1.5 mas")
    print("="*65 + "\n")
    
    # Construct the ADQL query
    adql_query = f"""
SELECT 
    source_id, ra, ra_error, dec, dec_error, 
    parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, 
    radial_velocity, radial_velocity_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
    ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr,
    dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr,
    parallax_pmra_corr, parallax_pmdec_corr,
    pmra_pmdec_corr
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {ra_past:.6f}, {dec_past:.6f}, {radius_deg})
)
AND parallax >= {plx_min:.6f} AND parallax <= {plx_max:.6f}
AND parallax_over_error > 5
AND pmra IS NOT NULL
AND pmdec IS NOT NULL
"""
    
    print("Copy the following ADQL block and paste it into the Gaia Archive:\n")
    print(adql_query.strip())

if __name__ == "__main__":
    TARGET_PULSAR = "J1124-5916"
    SEARCH_RADIUS_DEG = 1.0
    
    generate_query(target_id=TARGET_PULSAR, radius_deg=SEARCH_RADIUS_DEG)