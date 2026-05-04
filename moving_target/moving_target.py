import os, shutil
import pandas as pd
import numpy as np
from pygaia.astrometry.coordinates import EpochPropagation
import extinction 
from PIL import Image
from matplotlib.patches import Ellipse
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches



np.random.seed(42)

csv_file = 'stars.csv'
fits_file = 'cygnus.fits'

target_id = '1859461255653506176'
#target_id = '1858717401682424064'
#target_id = '1858675134890513152'

# TIME & SIMULATION PARAMETERS
tau_mean = 100280      
tau_error = 30020
tau_min = 70000
tau_max = 135000      
N = 1000               
#deg_from_center = 0.02  
number_sigma = 3


def calculate_overlap_2d(ra_cand, dec_cand, ra_target, dec_target, n_sigma, num_particles):
    """
    Calcula la probabilidad de intersección mutua entre dos nubes de puntos 2D 
    (Estrella y Púlsar) usando radios circulares, y devuelve la distancia entre sus centros.
    """
    # centros
    cand_ra_mean = np.mean(ra_cand)
    cand_dec_mean = np.mean(dec_cand)
    ns_ra_mean = np.mean(ra_target)
    ns_dec_mean = np.mean(dec_target)
    
    # desviaciones estándar y radios
    cand_ra_std = np.std(ra_cand) * np.cos(np.radians(cand_dec_mean))
    cand_dec_std = np.std(dec_cand)
    ns_ra_std = np.std(ra_target) * np.cos(np.radians(ns_dec_mean))
    ns_dec_std = np.std(dec_target)

    cand_radius = np.sqrt(cand_ra_std**2 + cand_dec_std**2) * n_sigma
    ns_radius = np.sqrt(ns_ra_std**2 + ns_dec_std**2) * n_sigma
    
    # fracción Púlsar
    dx_pulsar = (ra_target - cand_ra_mean) * np.cos(np.radians(cand_dec_mean))
    dy_pulsar = dec_target - cand_dec_mean
    frac_pulsar = np.sum(np.sqrt(dx_pulsar**2 + dy_pulsar**2) < cand_radius) / num_particles

    # fracción Estrella
    dx_cand = (ra_cand - ns_ra_mean) * np.cos(np.radians(ns_dec_mean))
    dy_cand = dec_cand - ns_dec_mean
    frac_star = np.sum(np.sqrt(dx_cand**2 + dy_cand**2) < ns_radius) / num_particles

    # probabilidad cruzada y distancia mínima
    prob = frac_pulsar * frac_star
    
    d_ra_center = (cand_ra_mean - ns_ra_mean) * np.cos(np.radians(ns_dec_mean))
    d_dec_center = (cand_dec_mean - ns_dec_mean)
    min_dist = np.sqrt(d_ra_center**2 + d_dec_center**2)

    return prob, min_dist, ns_radius


df = pd.read_csv(csv_file, dtype={'source_id': str})

df['radial_velocity'] = df['radial_velocity'].fillna(0.0)

print(f"Original stars: {len(df)}.")


# ASTROPHYSICAL FILTERING & EXTINCTION

df = df[(df['parallax'] > 1.190476) & (df['parallax'] < 3.333333)].copy()

# distance in parsecs and kiloparsecs
df['distance_pc'] = 1000.0 / df['parallax']
df['distance_kpc'] = df['distance_pc'] / 1000.0

# Kinematics basics
df['pm_total'] = np.sqrt(df['pmra']**2 + df['pmdec']**2)
df['v_transverse'] = 4.74 * (df['pm_total'] / df['parallax'])

av_data = np.loadtxt('Av_Cygnus_Loop.txt')
dist_kpc_profile = av_data[:, 0]
av_profile = av_data[:, 1]

# interpolate Av for each star's exact distance
df['A_V'] = np.interp(df['distance_kpc'], dist_kpc_profile, av_profile)

# effective wavelengths for Gaia DR3 from SVO 
# G = 5822.39 A, BP = 5035.75 A, RP = 7619.96 A
wave_gaia = np.array([5822.39, 5035.75, 7619.96])

# A_G, A_BP, and A_RP band by band (assuming standard Milky Way R_v = 3.1)
a_g, a_bp, a_rp = [], [], []
for av in df['A_V']:
    ext = extinction.fitzpatrick99(wave_gaia, av, 3.1)
    a_g.append(ext[0])
    a_bp.append(ext[1])
    a_rp.append(ext[2])
    
df['A_G'] = a_g
df['A_BP'] = a_bp
df['A_RP'] = a_rp

# intrinsic absolute magnitude (M_G)
df['abs_mag_g'] = df['phot_g_mean_mag'] - 5 * np.log10(df['distance_pc']) + 5 - df['A_G']

# intrinsic color (BP-RP)
bp_0 = df['phot_bp_mean_mag'] - df['A_BP']
rp_0 = df['phot_rp_mean_mag'] - df['A_RP']
df['bp_rp_color'] = bp_0 - rp_0

# error propagation
df['pm_total_error'] = np.sqrt((df['pmra'] * df['pmra_error'])**2 + (df['pmdec'] * df['pmdec_error'])**2) / df['pm_total']
relative_error_pm = df['pm_total_error'] / df['pm_total']
relative_error_px = df['parallax_error'] / df['parallax']
df['v_transverse_error'] = df['v_transverse'] * np.sqrt(relative_error_pm**2 + relative_error_px**2)

# final cut: Keep only massive, hot, young stars (Absolute Mag < 5, Color < 1)
#df = df[(df['abs_mag_g'] < 5) & (df['bp_rp_color'] < 1)]
print(f"Filtering complete. {len(df)} candidate stars remaining.")


# MOVING TARGET

print(f"\nSetting up the moving target: Star {target_id}")
if target_id not in df['source_id'].values:
    raise ValueError(f"Target star {target_id} not found in the filtered catalog! Check your filters.")

target_star = df[df['source_id'] == target_id].iloc[0]

ns_ra = target_star['ra']
ns_dec = target_star['dec']
ns_parallax = target_star['parallax']
ns_pmra = target_star['pmra']
ns_pmdec = target_star['pmdec']
ns_rv = target_star['radial_velocity']
ns_pmra_err = target_star['pmra_error']
ns_pmdec_err = target_star['pmdec_error']

print(f"Target parameters: PMRA={ns_pmra:.2f}, PMDEC={ns_pmdec:.2f}, RV={ns_rv:.2f}")

# MONTE CARLO 

print(f"\nStarting Monte Carlo simulation (N={N}) for {len(df)} stars...")

probabilities = []
min_distances = []
ep = EpochPropagation()

# neutron star data
np.random.seed(888)
ns_ra_arr = np.full(N, np.radians(ns_ra))
ns_dec_arr = np.full(N, np.radians(ns_dec))
ns_px_arr = np.full(N, ns_parallax)
ns_rv_arr = np.full(N, ns_rv)
ns_pmra_samples = np.random.normal(loc=ns_pmra, scale=ns_pmra_err, size=N)
ns_pmdec_samples = np.random.normal(loc=ns_pmdec, scale=ns_pmdec_err, size=N)

for i, row in df.iterrows():
    np.random.seed(i)
    
    # Gaussian samples for the candidate star kinematics and time
    pmra_samples = np.random.normal(loc=row['pmra'], scale=row['pmra_error'], size=N)
    pmdec_samples = np.random.normal(loc=row['pmdec'], scale=row['pmdec_error'], size=N)
    tau_samples = np.random.normal(loc=tau_mean, scale=tau_error, size=N)
    
    ra_rad_arr = np.full(N, np.radians(row['ra']))
    dec_rad_arr = np.full(N, np.radians(row['dec']))
    px_arr = np.full(N, row['parallax'])
    rv_arr = np.full(N, row['radial_velocity'])

    # Propagate candidate to the past
    ra_past_rad, dec_past_rad = ep.propagate_pos(
        ra_rad_arr, dec_rad_arr, px_arr, 
        pmra_samples, pmdec_samples, rv_arr, 2016.0, 2016.0 - tau_samples
    )
    ra_past = np.degrees(ra_past_rad)
    dec_past = np.degrees(dec_past_rad)
    
    ns_ra_past_rad, ns_dec_past_rad = ep.propagate_pos(
        ns_ra_arr, ns_dec_arr, ns_px_arr, 
        ns_pmra_samples, ns_pmdec_samples, ns_rv_arr, 2016.0, 2016.0 - tau_samples
    )
    ns_ra_past = np.degrees(ns_ra_past_rad)
    ns_dec_past = np.degrees(ns_dec_past_rad)

    prob, min_dist, _ = calculate_overlap_2d(ra_past, dec_past, ns_ra_past, ns_dec_past, number_sigma, N)
    
    probabilities.append(prob)
    min_distances.append(min_dist)

df['crossing_probability'] = probabilities
df['min_distance_deg'] = min_distances


# RESULTS

candidates = df[df['crossing_probability'] > 0].copy()

# PROBABILITY

print("\nCalculating overlap probability at each time step for top candidates...")

eval_times = np.arange(tau_min, tau_max, 1000)

# pre-calcular la historia temporal del Púlsar
pulsar_history_ra = {}
pulsar_history_dec = {}

for t in eval_times:
    ns_ra_past_rad, ns_dec_past_rad = ep.propagate_pos(
        ns_ra_arr, ns_dec_arr, ns_px_arr, 
        ns_pmra_samples, ns_pmdec_samples, ns_rv_arr, 2016.0, 2016.0 - t
    )
    pulsar_history_ra[t] = np.degrees(ns_ra_past_rad)
    pulsar_history_dec[t] = np.degrees(ns_dec_past_rad)

prob_over_time = {}
peak_probabilities = []
peak_ages = []

for idx, (index, row) in enumerate(candidates.iterrows()):
    np.random.seed(index)
    
    # Nube Monte Carlo de la candidata
    pmra_samples = np.random.normal(loc=row['pmra'], scale=row['pmra_error'], size=N)
    pmdec_samples = np.random.normal(loc=row['pmdec'], scale=row['pmdec_error'], size=N)
    
    ra_rad_arr = np.full(N, np.radians(row['ra']))
    dec_rad_arr = np.full(N, np.radians(row['dec']))
    px_arr = np.full(N, row['parallax'])
    rv_arr = np.full(N, row['radial_velocity'])
    
    time_probs = []
    max_prob_for_star = 0.0
    age_at_max = 0
    
    # vamos a cada step de tiempo específico (t)
    for t in eval_times:
        # propagar Candidata
        ra_past_rad, dec_past_rad = ep.propagate_pos(
            ra_rad_arr, dec_rad_arr, px_arr, 
            pmra_samples, pmdec_samples, rv_arr, 2016.0, 2016.0 - t
        )
        ra_past = np.degrees(ra_past_rad)
        dec_past = np.degrees(dec_past_rad)
        
        ns_ra_past = pulsar_history_ra[t]
        ns_dec_past = pulsar_history_dec[t]

        prob, min_dist, ns_radius_3sigma = calculate_overlap_2d(ra_past, dec_past, ns_ra_past, ns_dec_past, number_sigma, N)
        
        time_probs.append(prob)

        if prob > max_prob_for_star:
            max_prob_for_star = prob
            age_at_max = t
        
    prob_over_time[row['source_id']] = time_probs
    peak_probabilities.append(max_prob_for_star)
    peak_ages.append(age_at_max)

candidates['peak_probability'] = peak_probabilities
candidates['kinematic_age'] = peak_ages

candidates = candidates.sort_values(by=['peak_probability', 'min_distance_deg'], ascending=[False, True])
top_candidates = candidates.head(10)

print("\nTOP CANDIDATES")
columns_to_print = ['source_id', 'peak_probability', 'kinematic_age', 'min_distance_deg', 'pm_total', 'v_transverse']
print(candidates[columns_to_print].head(10))


colors = cm.rainbow(np.linspace(0, 1, len(top_candidates)))
np.random.seed(99) 
np.random.shuffle(colors)


plt.figure(figsize=(10, 6))

for idx, (index, row) in enumerate(top_candidates.iterrows()):
    source = row['source_id']
    probs = prob_over_time[source]
    color = colors[idx] 
    
    if max(probs) > 0:
        plt.plot(eval_times / 1000.0, probs, linewidth=2, marker='.', color=color, label=f"{idx + 1}: {source}")

plt.axvline(x=tau_mean/1000.0, color='black', linestyle='--', linewidth=2, label='Supernova Est. Age (100 kyr)')

plt.title('Kinematic Intersection Probability over Time', fontsize=14, fontweight='bold')
plt.xlabel('Time into the past (kilo-years)', fontsize=12)
plt.ylabel('Overlap Probability (Hits / N)', fontsize=12)
plt.xlim(70, 135)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show(block=False)


# EXPORT DETAILED CANDIDATES TABLE 

print("\nGenerating professional Excel report with two sheets...")

time_columns = [f"{int(t/1000)} kyr" for t in eval_times]

cols_info = [
    'source_id', 'peak_probability', 'kinematic_age', 'crossing_probability', 'min_distance_deg',
    'ra', 'dec', 'distance_pc', 'v_transverse', 'radial_velocity',
    'abs_mag_g', 'bp_rp_color', 'A_V'
]
df_info = top_candidates[cols_info].copy()

df_info.rename(columns={
    'source_id': 'Gaia DR3 Source ID',
    'peak_probability': 'Peak Prob.',
    'kinematic_age': 'Kinematic age',
    'crossing_probability': 'Global Hit Prob.',
    'min_distance_deg': 'Min Dist (°)',
    'ra': 'RA (deg)',
    'dec': 'DEC (deg)',
    'distance_pc': 'Dist (pc)',
    'v_transverse': 'V_T (km/s)',
    'radial_velocity': 'RV (km/s)',
    'abs_mag_g': 'Abs Mag (Mg)',
    'bp_rp_color': 'Color (BP-RP)0',
    'A_V': 'Extinction (Av)'
}, inplace=True)

prob_data = {'Gaia DR3 Source ID': top_candidates['source_id'].values}

for t_idx, t_val in enumerate(eval_times):
    col_name = f"{int(t_val/1000)} kyr"
    prob_data[col_name] = [prob_over_time[sid][t_idx] for sid in top_candidates['source_id']]

df_probs = pd.DataFrame(prob_data)

excel_filename = 'Top_Candidates_Analysis.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df_info.to_excel(writer, sheet_name='Astrophysical_Details', index=False)
    df_probs.to_excel(writer, sheet_name='Time_Series_Probs', index=False)

print(f"Excel file created: {excel_filename}")

# DRAW 

image = fits.open(fits_file)
data = image[0].data 
w = WCS(image[0].header)

def get_pixel(ra_val, dec_val):
    """Converts RA/DEC to pixel coordinates based on the FITS WCS."""
    c = SkyCoord(ra_val * u.deg, dec_val * u.deg, frame='icrs')
    return w.world_to_pixel(SkyCoord(c.galactic.l.value * u.deg, c.galactic.b.value * u.deg, frame='galactic'))

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1, 1, 1, projection=w)

vmin_auto = np.nanpercentile(data, 1)
vmax_auto = np.nanpercentile(data, 99)
ax1.imshow(data, vmin=vmin_auto, vmax=vmax_auto, cmap='Greys', origin='lower')


# calculate target's exact past position
ns_ra_cen_rad, ns_dec_cen_rad = ep.propagate_pos(
    np.radians(ns_ra), np.radians(ns_dec), ns_parallax, 
    ns_pmra, ns_pmdec, ns_rv, 2016.0, 2016.0 - tau_mean
)

target_past_x, target_past_y = get_pixel(np.degrees(ns_ra_cen_rad), np.degrees(ns_dec_cen_rad))
target_past_x, target_past_y = float(target_past_x), float(target_past_y)

# Yellow dot for the target's past position
ax1.plot(target_past_x, target_past_y, '.', color='yellow', markersize=15, markeredgecolor='black', zorder=10)

np.random.seed(888)
draw_pmra_samples = np.random.normal(loc=ns_pmra, scale=ns_pmra_err, size=N)
draw_pmdec_samples = np.random.normal(loc=ns_pmdec, scale=ns_pmdec_err, size=N)

ns_ra_rad_arr = np.full(N, np.radians(ns_ra))
ns_dec_rad_arr = np.full(N, np.radians(ns_dec))
ns_px_arr = np.full(N, ns_parallax)
ns_rv_arr = np.full(N, ns_rv)

draw_ra_past_rad, draw_dec_past_rad = ep.propagate_pos(
    ns_ra_rad_arr, ns_dec_rad_arr, ns_px_arr, 
    draw_pmra_samples, draw_pmdec_samples, ns_rv_arr, 2016.0, 2016.0 - tau_mean
)

_, _, draw_radius_3sigma = calculate_overlap_2d(
    np.degrees(draw_ra_past_rad), np.degrees(draw_dec_past_rad), 
    np.degrees(draw_ra_past_rad), np.degrees(draw_dec_past_rad), 
    number_sigma, N
)

# Red dashed circle for the impact area
pixel_scale = proj_plane_pixel_scales(w)[0]
radius_pixels = draw_radius_3sigma / pixel_scale
target_circle = patches.Circle((target_past_x, target_past_y), radius=radius_pixels, 
                               edgecolor='red', facecolor='none', linestyle='--', 
                               linewidth=2, alpha=0.8, zorder=4)
ax1.add_patch(target_circle)



for idx, (index, row) in enumerate(top_candidates.iterrows()):
    color = colors[idx]
    
    ra, dec = row['ra'], row['dec']
    pmra, e_pmra = row['pmra'], row['pmra_error']
    pmdec, e_pmdec = row['pmdec'], row['pmdec_error']

    np.random.seed(index)
    
    pmra_samples = np.random.normal(loc=pmra, scale=e_pmra, size=N)
    pmdec_samples = np.random.normal(loc=pmdec, scale=e_pmdec, size=N)
    tau_samples = np.random.normal(loc=tau_mean, scale=tau_error, size=N)

    ra_rad_arr = np.full(N, np.radians(ra))
    dec_rad_arr = np.full(N, np.radians(dec))
    px_arr = np.full(N, row['parallax'])
    rv_arr = np.full(N, row['radial_velocity'])

    # Monte Carlo (Past)
    ra_past_rad, dec_past_rad = ep.propagate_pos(
        ra_rad_arr, dec_rad_arr, px_arr, 
        pmra_samples, pmdec_samples, rv_arr, 2016.0, 2016.0 - tau_samples
    )
    ra_past_samples = np.degrees(ra_past_rad)
    dec_past_samples = np.degrees(dec_past_rad)
    
    # Exact present
    X_num, Y_num = get_pixel(ra, dec)
    X_float, Y_float = float(X_num), float(Y_num)
    
    # Exact past (Central line)
    ra_cen_rad, dec_cen_rad = ep.propagate_pos(
        np.radians(ra), np.radians(dec), row['parallax'], 
        pmra, pmdec, row['radial_velocity'], 2016.0, 2016.0 - tau_mean
    )
    x_central, y_central = get_pixel(np.degrees(ra_cen_rad), np.degrees(dec_cen_rad))
    x_central, y_central = float(x_central), float(y_central)

    # Draw Monte Carlo (Transparent lines)
    for i in range(N):
        x_pix, y_pix = get_pixel(ra_past_samples[i], dec_past_samples[i])
        ax1.plot([float(x_pix), X_float], [float(y_pix), Y_float], '-', color=color, alpha=0.03, zorder=1)

    # Draw central trajectory
    ax1.plot([x_central, X_float], [y_central, Y_float], '-', color=color, linewidth=2, zorder=5)
    ax1.plot(x_central, y_central, 'X', color=color, markersize=8, markeredgecolor='black', zorder=6)
    
    # Label text
    label_text = f'{idx+1}: {row["source_id"]}'
    if row["source_id"] == target_id:
        label_text += ' *'
        
    ax1.plot(X_float, Y_float, 'o', color=color, markersize=8, markeredgecolor='white', label=label_text, zorder=7)
    ax1.text(X_float + 0.05, Y_float, str(idx + 1), color='black', fontsize=8, fontweight='bold', zorder=7)
    


center_ra, center_dec = 312.75, 30.667 
center_x, center_y = get_pixel(center_ra, center_dec)
center_x, center_y = float(center_x), float(center_y)

ax1.set_xlim([center_x - 100, center_x + 100])
ax1.set_ylim([center_y - 100, center_y + 100])

plt.title('Kinematic Intersection (Moving Target)', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=8)
plt.show(block=False)


# HR DIAGRAM 

print("\nGenerating HR Diagram...")

df_hr = df.dropna(subset=['bp_rp_color', 'abs_mag_g'])

plt.figure(figsize=(10, 7))

scat = plt.scatter(
    df_hr['bp_rp_color'], df_hr['abs_mag_g'], 
    c=df_hr['bp_rp_color'], cmap='coolwarm',       
    s=25, alpha=1, zorder=1,
    vmin=-0.5, vmax=3.5   
)

plt.vlines(x=1.0, ymin=-5, ymax=5.0, color='red', linestyle='--', linewidth=1.5, label='Color cut (BP-RP < 1)')
plt.hlines(y=5.0, xmin=-1, xmax=1.0, color='green', linestyle='--', linewidth=1.5, label='Mag cut (M_G < 5)')

cb = plt.colorbar(scat)
cb.set_label('Temperature (Color BP-RP)')

# Highlight the Top Candidates & Target Star
for idx, (_, row) in enumerate(top_candidates.iterrows()):
    star_data_hr = df_hr[df_hr['source_id'] == row['source_id']]
    
    if not star_data_hr.empty:
        col = star_data_hr['bp_rp_color'].values[0]
        m_g = star_data_hr['abs_mag_g'].values[0]
        color = colors[idx]
        
        label_text = f'{idx+1}: {row["source_id"]}'
        
        if row["source_id"] == target_id:
            label_text += ' *'
            plt.scatter(col, m_g, color='yellow', edgecolor='black', s=400, marker='.', zorder=6, label=label_text)
        else:
            plt.scatter(col, m_g, color=color, edgecolor='black', s=200, marker='.', zorder=5, label=label_text)
        
        plt.text(col + 0.05, m_g, str(idx + 1), color='black', fontsize=10, fontweight='bold', zorder=7)

plt.gca().invert_yaxis()
plt.title('HR Diagram', fontsize=14, fontweight='bold')
plt.xlabel('Intrinsic Color $(BP - RP)_0$', fontsize=12)
plt.ylabel('Absolute Magnitude $M_G$', fontsize=12)
plt.xlim(-1, 4)
plt.ylim(15, -5)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9, framealpha=0.9, edgecolor='black')

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show(block=False)


# -------  GIF  -----------

output_folder = 'runaway_frames'
gif_filename = 'runaway_evolution.gif'

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

time_centers = np.arange(tau_min, tau_max, 5000)
time_window = 10000

print("\nCreating the GIF...")

# MONTE CARLO FOR TARGET STAR
np.random.seed(888)
ns_pmra_err = target_star['pmra_error']
ns_pmdec_err = target_star['pmdec_error']

ns_pmra_samples = np.random.normal(loc=ns_pmra, scale=ns_pmra_err, size=N)
ns_pmdec_samples = np.random.normal(loc=ns_pmdec, scale=ns_pmdec_err, size=N)

ns_ra_rad_arr = np.full(N, np.radians(ns_ra))
ns_dec_rad_arr = np.full(N, np.radians(ns_dec))
ns_px_arr = np.full(N, ns_parallax)
ns_rv_arr = np.full(N, ns_rv)

# static position
t_max = np.max(time_centers)
ns_ra_max_rad, ns_dec_max_rad = ep.propagate_pos(
    np.radians(ns_ra), np.radians(ns_dec), ns_parallax, 
    ns_pmra, ns_pmdec, ns_rv, 2016.0, 2016.0 - t_max
)
ns_ra_max = np.degrees(ns_ra_max_rad)
ns_dec_max = np.degrees(ns_dec_max_rad)

center_ra_static = (ns_ra + ns_ra_max) / 2.0
center_dec_static = (ns_dec + ns_dec_max) / 2.0

ra_span = abs(ns_ra - ns_ra_max)
dec_span = abs(ns_dec - ns_dec_max)
zoom_window_static = max(ra_span, dec_span) / 2.0 + 0.1  

for frame_idx, t_mid in enumerate(time_centers):
    t_kyr = t_mid / 1000.0
    print(f"  -> Processing frame: {t_kyr:.0f} kyr ago")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    
    ax1.scatter(df['ra'], df['dec'], s=1, color='lightgray', alpha=0.3)
    ax2.scatter(df['ra'], df['dec'], s=1, color='lightgray', alpha=0.3)
    
    ns_ra_cloud_rad, ns_dec_cloud_rad = ep.propagate_pos(
        ns_ra_rad_arr, ns_dec_rad_arr, ns_px_arr, 
        ns_pmra_samples, ns_pmdec_samples, ns_rv_arr, 
        2016.0, 2016.0 - t_mid
    )
    ns_ra_cloud = np.degrees(ns_ra_cloud_rad)
    ns_dec_cloud = np.degrees(ns_dec_cloud_rad)
    
    ns_ra_mean = np.mean(ns_ra_cloud)
    ns_dec_mean = np.mean(ns_dec_cloud)
    ns_ra_sigma = np.std(ns_ra_cloud)
    ns_dec_sigma = np.std(ns_dec_cloud)
    
    # Dibujamos toda la línea de la trayectoria del target para que sirva de guía
    ax1.plot([ns_ra, ns_ra_max], [ns_dec, ns_dec_max], '--', color='yellow', alpha=0.4, linewidth=1)
    
    # Posición actual del Target
    target_label = f"{target_id} *"
    ax1.plot(ns_ra_mean, ns_dec_mean, 'o', color='yellow', markersize=10, markeredgecolor='black', zorder=10, label=target_label)
    ax2.plot(ns_ra_mean, ns_dec_mean, 'o', color='yellow', markersize=10, markeredgecolor='black', zorder=10)
    
    ax2.plot(ns_ra_cloud, ns_dec_cloud, '.', color='yellow', alpha=0.05, markersize=3, zorder=6)
    
    #w_elipse = max(6 * ns_ra_sigma, 0.02)
    #h_elipse = max(6 * ns_dec_sigma, 0.02)

    ellipse1 = Ellipse((ns_ra_mean, ns_dec_mean), width=number_sigma*2*ns_ra_sigma, height=number_sigma*2*ns_dec_sigma, 
                       edgecolor='red', facecolor='none', linestyle='--', linewidth=2, zorder=4)
    ellipse2 = Ellipse((ns_ra_mean, ns_dec_mean), width=number_sigma*2*ns_ra_sigma, height=number_sigma*2*ns_dec_sigma, 
                       edgecolor='red', facecolor='none', linestyle='--', linewidth=2, zorder=4)
    ax1.add_patch(ellipse1)
    ax2.add_patch(ellipse2)

    ax1.set_title(f'Exact Trajectories\n Time: {t_kyr:.0f} kyr ago', fontsize=14, fontweight='bold')
    
    t_start = max(0, t_mid - (time_window / 2.0))
    t_end = t_mid + (time_window / 2.0)
    bin_label = f'{t_start/1000:.1f} to {t_end/1000:.1f} kyr ago'
    ax2.set_title(f'Monte Carlo Cloud\n Time: {bin_label}', fontsize=14, fontweight='bold')

    for idx, (index, row) in enumerate(top_candidates.iterrows()):
        color = colors[idx]
        label_text = f'{idx+1}: {row["source_id"]}'
        
        # --- LEFT ---
        ra_mid_rad, dec_mid_rad = ep.propagate_pos(
            np.radians(row['ra']), np.radians(row['dec']), row['parallax'], 
            row['pmra'], row['pmdec'], row['radial_velocity'], 2016.0, 2016.0 - t_mid
        )
        ra_mid = np.degrees(ra_mid_rad)
        dec_mid = np.degrees(dec_mid_rad)
        
        ax1.plot([row['ra'], ra_mid], [row['dec'], dec_mid], '-', color=color, linewidth=2, alpha=0.5)
        ax1.plot(ra_mid, dec_mid, 'o', color=color, markersize=8, markeredgecolor='black', zorder=9, label=label_text)
        
        # --- RIGHT ---
        np.random.seed(index) 
        pmra_samples = np.random.normal(loc=row['pmra'], scale=row['pmra_error'], size=N)
        pmdec_samples = np.random.normal(loc=row['pmdec'], scale=row['pmdec_error'], size=N)
        
        times_cont = np.linspace(t_start, t_end, 50)
        
        ra_rad_50 = np.full(50, np.radians(row['ra']))
        dec_rad_50 = np.full(50, np.radians(row['dec']))
        px_50 = np.full(50, row['parallax'])
        rv_50 = np.full(50, row['radial_velocity'])
        
        for i in range(N):
            pmra_50 = np.full(50, pmra_samples[i])
            pmdec_50 = np.full(50, pmdec_samples[i])
            
            ra_mc_rad, dec_mc_rad = ep.propagate_pos(
                ra_rad_50, dec_rad_50, px_50, 
                pmra_50, pmdec_50, rv_50, 2016.0, 2016.0 - times_cont
            )
            ax2.plot(np.degrees(ra_mc_rad), np.degrees(dec_mc_rad), '-', color=color, alpha=0.03, linewidth=1, zorder=7)
            
    ax1.set_xlim(center_ra_static + zoom_window_static, center_ra_static - zoom_window_static) 
    ax1.set_ylim(center_dec_static - zoom_window_static, center_dec_static + zoom_window_static)
    
    ax1.set_xlabel('Right Ascension (deg)')
    ax1.set_ylabel('Declination (deg)')
    ax2.set_xlabel('Right Ascension (deg)')
    
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    ax1.legend(loc='lower left', fontsize=8, framealpha=0.8, edgecolor='black')
    
    plt.tight_layout()
    filename = f'frame_{t_kyr:03.0f}.png'
    plt.savefig(os.path.join(output_folder, filename), dpi=150)
    plt.close()

print("Compiling GIF...")
files = [f for f in os.listdir(output_folder) if f.endswith('.png')]
files.sort()

if files:
    frames = []
    for file in files:
        route = os.path.join(output_folder, file)
        frames.append(Image.open(route))
        
    frames[0].save(
        gif_filename,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=300,  
        loop=0         
    )
    print(f"GIF created: {gif_filename}")

plt.show()