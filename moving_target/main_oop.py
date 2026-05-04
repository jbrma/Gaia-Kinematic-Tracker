import pandas as pd
import numpy as np
import extinction
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from pygaia.astrometry.coordinates import EpochPropagation

class StarCatalog:
    def __init__(self, csv_file, av_file='Av_Cygnus_Loop.txt'):
        """Initializes the catalog by loading the raw data."""
        self.csv_file = csv_file
        self.av_file = av_file
        
        self.df = pd.read_csv(self.csv_file, dtype={'source_id': str})
        self.df['radial_velocity'] = self.df['radial_velocity'].fillna(0.0)
        self.df['radial_velocity_error'] = self.df['radial_velocity_error'].fillna(0.0)
        
        print(f"Original stars: {len(self.df)}.")

    def apply_filters(self):
        """Applies astrophysical filters, extinction, and kinematics."""
        # ASTROPHYSICAL FILTERING & EXTINCTION

        self.df = self.df[(self.df['parallax'] > 1.190476) & (self.df['parallax'] < 3.333333)].copy()

        # distance in parsecs and kiloparsecs
        self.df['distance_pc'] = 1000.0 / self.df['parallax']
        self.df['distance_kpc'] = self.df['distance_pc'] / 1000.0

        # Kinematics basics
        self.df['pm_total'] = np.sqrt(self.df['pmra']**2 + self.df['pmdec']**2)
        self.df['v_transverse'] = 4.74 * (self.df['pm_total'] / self.df['parallax'])

        av_data = np.loadtxt(self.av_file)
        dist_kpc_profile = av_data[:, 0]
        av_profile = av_data[:, 1]

        # interpolate Av for each star's exact distance
        self.df['A_V'] = np.interp(self.df['distance_kpc'], dist_kpc_profile, av_profile)

        # effective wavelengths for Gaia DR3 from SVO 
        # G = 5822.39 A, BP = 5035.75 A, RP = 7619.96 A
        wave_gaia = np.array([5822.39, 5035.75, 7619.96])

        # A_G, A_BP, and A_RP band by band (assuming standard Milky Way R_v = 3.1)
        a_g, a_bp, a_rp = [], [], []
        for av in self.df['A_V']:
            ext = extinction.fitzpatrick99(wave_gaia, av, 3.1)
            a_g.append(ext[0])
            a_bp.append(ext[1])
            a_rp.append(ext[2])
            
        self.df['A_G'] = a_g
        self.df['A_BP'] = a_bp
        self.df['A_RP'] = a_rp

        # intrinsic absolute magnitude (M_G)
        self.df['abs_mag_g'] = self.df['phot_g_mean_mag'] - 5 * np.log10(self.df['distance_pc']) + 5 - self.df['A_G']

        # intrinsic color (BP-RP)
        bp_0 = self.df['phot_bp_mean_mag'] - self.df['A_BP']
        rp_0 = self.df['phot_rp_mean_mag'] - self.df['A_RP']
        self.df['bp_rp_color'] = bp_0 - rp_0

        # error propagation
        self.df['pm_total_error'] = np.sqrt((self.df['pmra'] * self.df['pmra_error'])**2 + (self.df['pmdec'] * self.df['pmdec_error'])**2) / self.df['pm_total']
        relative_error_pm = self.df['pm_total_error'] / self.df['pm_total']
        relative_error_px = self.df['parallax_error'] / self.df['parallax']
        self.df['v_transverse_error'] = self.df['v_transverse'] * np.sqrt(relative_error_pm**2 + relative_error_px**2)

        # final cut: Keep only massive, hot, young stars (Absolute Mag < 5, Color < 1)
        #self.df = self.df[(self.df['abs_mag_g'] < 5) & (self.df['bp_rp_color'] < 1)]
        print(f"Filtering complete. {len(self.df)} candidate stars remaining.")
        
        return self.df

    def get_target_star(self, target_id):
        """Extracts the moving target star parameters."""
        print(f"\nSetting up the moving target: Star {target_id}")
        if target_id not in self.df['source_id'].values:
            raise ValueError(f"Target star {target_id} not found in the filtered catalog! Check your filters.")
        
        target_star = self.df[self.df['source_id'] == target_id].iloc[0]
        
        ns_pmra = target_star['pmra']
        ns_pmdec = target_star['pmdec']
        ns_rv = target_star['radial_velocity']
        
        print(f"Pulsar parameters: PMRA={ns_pmra:.2f}, PMDEC={ns_pmdec:.2f}, RV={ns_rv:.2f}")
        
        return target_star


class KinematicSimulator:
    def __init__(self, target_star, tau_params, sim_params):
        """Initializes the simulation parameters and the moving target."""
        self.target = target_star
        self.tau_mean = tau_params['mean']
        self.tau_error = tau_params['error']
        self.tau_min = tau_params['min']
        self.tau_max = tau_params['max']
        self.N = sim_params['N']
        self.n_sigma = sim_params['sigma']
        
        self.ep = EpochPropagation()
        self._init_target_cloud()

    def _init_target_cloud(self):
        """Pre-calculates the static Monte Carlo base cloud for the target star."""
        np.random.seed(888) # Fixed seed for the pulsar
        self.ns_ra_arr = np.full(self.N, np.radians(self.target['ra']))
        self.ns_dec_arr = np.full(self.N, np.radians(self.target['dec']))
        self.ns_px_arr = np.full(self.N, self.target['parallax'])
        self.ns_rv_samples = np.random.normal(loc=self.target['radial_velocity'], scale=self.target['radial_velocity_error'], size=self.N)
        
        self.ns_pmra_samples = np.random.normal(loc=self.target['pmra'], scale=self.target['pmra_error'], size=self.N)
        self.ns_pmdec_samples = np.random.normal(loc=self.target['pmdec'], scale=self.target['pmdec_error'], size=self.N)


    def calculate_overlap_2d(self, ra_cand, dec_cand, ra_target, dec_target):
        """Calculates mutual intersection probability using 2D Circles (Original Method)."""
        cand_ra_mean, cand_dec_mean = np.mean(ra_cand), np.mean(dec_cand)
        ns_ra_mean, ns_dec_mean = np.mean(ra_target), np.mean(dec_target)

        cand_ra_std = np.std(ra_cand) * np.cos(np.radians(cand_dec_mean))
        cand_dec_std = np.std(dec_cand)
        ns_ra_std = np.std(ra_target) * np.cos(np.radians(ns_dec_mean))
        ns_dec_std = np.std(dec_target)

        cand_radius = np.sqrt(cand_ra_std**2 + cand_dec_std**2) * self.n_sigma
        ns_radius = np.sqrt(ns_ra_std**2 + ns_dec_std**2) * self.n_sigma

        # pulsar fraction
        dx_pulsar = (ra_target - cand_ra_mean) * np.cos(np.radians(cand_dec_mean))
        dy_pulsar = dec_target - cand_dec_mean
        frac_pulsar = np.sum(np.sqrt(dx_pulsar**2 + dy_pulsar**2) < cand_radius) / self.N

        # star fraction
        dx_star = (ra_cand - ns_ra_mean) * np.cos(np.radians(ns_dec_mean))
        dy_star = dec_cand - ns_dec_mean
        frac_star = np.sum(np.sqrt(dx_star**2 + dy_star**2) < ns_radius) / self.N

        prob = frac_pulsar * frac_star
        min_dist = np.sqrt(((cand_ra_mean - ns_ra_mean) * np.cos(np.radians(ns_dec_mean)))**2 + (cand_dec_mean - ns_dec_mean)**2)

        return prob, min_dist
    
    def get_ellipse_tools(self, x_points, y_points, n_sigma):
        """Math helper for 2D Ellipses (Mahalanobis)."""
        cov = np.cov(x_points, y_points)
        inv_cov = np.linalg.pinv(cov)
        evals, evecs = np.linalg.eigh(cov)
        
        width = 2 * n_sigma * np.sqrt(evals[1])
        height = 2 * n_sigma * np.sqrt(evals[0])
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        
        return inv_cov, width, height, angle

    def calculate_overlap_2d_ellipse(self, ra_cand, dec_cand, ra_target, dec_target):
        """ Calculates the overlap probability using rotated ellipses. """
        # Point cloud means
        cand_ra_mean, cand_dec_mean = np.mean(ra_cand), np.mean(dec_cand)
        ns_ra_mean, ns_dec_mean = np.mean(ra_target), np.mean(dec_target)

        # PULSAR FRACTION (How many Pulsar points fall inside the Star's ellipse?)
        # Center points and correct for spherical projection
        dx_p = (ra_target - cand_ra_mean) * np.cos(np.radians(cand_dec_mean))
        dy_p = dec_target - cand_dec_mean
        
        # Star's own points to calculate its shape and tilt
        star_x = (ra_cand - cand_ra_mean) * np.cos(np.radians(cand_dec_mean))
        star_y = dec_cand - cand_dec_mean
        
        # Get star's rotation angle and semi-axes using covariance
        cov_s = np.cov(star_x, star_y)
        evals_s, evecs_s = np.linalg.eigh(cov_s)
        angle_s = np.arctan2(evecs_s[1, 1], evecs_s[0, 1])
        a_s = np.sqrt(evals_s[1]) * self.n_sigma # Semi-major axis
        b_s = np.sqrt(evals_s[0]) * self.n_sigma # Semi-minor axis
        
        # Rotate the Pulsar points to "straighten" the Star's ellipse
        dx_p_rot = dx_p * np.cos(angle_s) + dy_p * np.sin(angle_s)
        dy_p_rot = -dx_p * np.sin(angle_s) + dy_p * np.cos(angle_s)
        
        # Count how many points fall inside using the standard ellipse equation
        frac_pulsar = np.sum((dx_p_rot / a_s)**2 + (dy_p_rot / b_s)**2 < 1.0) / self.N

        # STAR FRACTION (How many Star points fall inside the Pulsar's ellipse)
        # Center points and correct for spherical projection
        dx_s = (ra_cand - ns_ra_mean) * np.cos(np.radians(ns_dec_mean))
        dy_s = dec_cand - ns_dec_mean
        
        # Pulsar's own points to calculate its shape and tilt
        pulsar_x = (ra_target - ns_ra_mean) * np.cos(np.radians(ns_dec_mean))
        pulsar_y = dec_target - ns_dec_mean
        
        # Get pulsar's rotation angle and semi-axes using covariance
        cov_p = np.cov(pulsar_x, pulsar_y)
        evals_p, evecs_p = np.linalg.eigh(cov_p)
        angle_p = np.arctan2(evecs_p[1, 1], evecs_p[0, 1])
        a_p = np.sqrt(evals_p[1]) * self.n_sigma
        b_p = np.sqrt(evals_p[0]) * self.n_sigma
        
        # Rotate the Star points to "straighten" the Pulsar's ellipse
        dx_s_rot = dx_s * np.cos(angle_p) + dy_s * np.sin(angle_p)
        dy_s_rot = -dx_s * np.sin(angle_p) + dy_s * np.cos(angle_p)
        
        # Count how many points fall inside
        frac_star = np.sum((dx_s_rot / a_p)**2 + (dy_s_rot / b_p)**2 < 1.0) / self.N

        # Final probability and minimum distance
        prob = frac_pulsar * frac_star
        min_dist = np.sqrt(((cand_ra_mean - ns_ra_mean) * np.cos(np.radians(ns_dec_mean)))**2 + (cand_dec_mean - ns_dec_mean)**2)

        return prob, min_dist
    
        """Calculates mutual intersection probability using 2D Ellipses.
        cand_ra_mean, cand_dec_mean = np.mean(ra_cand), np.mean(dec_cand)
        ns_ra_mean, ns_dec_mean = np.mean(ra_target), np.mean(dec_target)

        cand_x = (ra_cand - cand_ra_mean) * np.cos(np.radians(cand_dec_mean))
        cand_y = dec_cand - cand_dec_mean
        ns_x = (ra_target - ns_ra_mean) * np.cos(np.radians(ns_dec_mean))
        ns_y = dec_target - ns_dec_mean

        cand_inv_cov, _, _, _ = self.get_ellipse_tools(cand_x, cand_y, self.n_sigma)
        ns_inv_cov, _, _, _ = self.get_ellipse_tools(ns_x, ns_y, self.n_sigma)

        # pulsar fraction
        dx_pulsar = (ra_target - cand_ra_mean) * np.cos(np.radians(cand_dec_mean))
        dy_pulsar = dec_target - cand_dec_mean
        dist_sq_pulsar = (dx_pulsar**2 * cand_inv_cov[0, 0] + 2 * dx_pulsar * dy_pulsar * cand_inv_cov[0, 1] + dy_pulsar**2 * cand_inv_cov[1, 1])
        frac_pulsar = np.sum(dist_sq_pulsar < self.n_sigma**2) / self.N

        # star fraction
        dx_star = (ra_cand - ns_ra_mean) * np.cos(np.radians(ns_dec_mean))
        dy_star = dec_cand - ns_dec_mean
        dist_sq_star = (dx_star**2 * ns_inv_cov[0, 0] + 2 * dx_star * dy_star * ns_inv_cov[0, 1] + dy_star**2 * ns_inv_cov[1, 1])
        frac_star = np.sum(dist_sq_star < self.n_sigma**2) / self.N

        prob = frac_pulsar * frac_star
        min_dist = np.sqrt(((cand_ra_mean - ns_ra_mean) * np.cos(np.radians(ns_dec_mean)))**2 + (cand_dec_mean - ns_dec_mean)**2)

        return prob, min_dist
        """


    def run_global_scan(self, df):
        """Runs the global probability scan for all stars."""
        print(f"\nStarting global Monte Carlo scan (N={self.N}) for {len(df)} stars...")
        probs, dists = [], []
        
        for i, row in df.iterrows():
            np.random.seed(i)
            pmra_samples = np.random.normal(loc=row['pmra'], scale=row['pmra_error'], size=self.N)
            pmdec_samples = np.random.normal(loc=row['pmdec'], scale=row['pmdec_error'], size=self.N)
            tau_samples = np.random.normal(loc=self.tau_mean, scale=self.tau_error, size=self.N)
            cand_rv_samples = np.random.normal(row['radial_velocity'], row['radial_velocity_error'], self.N)

            ra_rad_arr = np.full(self.N, np.radians(row['ra']))
            dec_rad_arr = np.full(self.N, np.radians(row['dec']))
            px_arr = np.full(self.N, row['parallax'])


            ra_past_rad, dec_past_rad = self.ep.propagate_pos(
                ra_rad_arr, dec_rad_arr, px_arr, 
                pmra_samples, pmdec_samples, cand_rv_samples, 2016.0, 2016.0 - tau_samples
            )
            
            ns_ra_past_rad, ns_dec_past_rad = self.ep.propagate_pos(
                self.ns_ra_arr, self.ns_dec_arr, self.ns_px_arr, 
                self.ns_pmra_samples, self.ns_pmdec_samples, self.ns_rv_samples, 2016.0, 2016.0 - tau_samples
            )

            prob, dist = self.calculate_overlap_2d_ellipse(
                np.degrees(ra_past_rad), np.degrees(dec_past_rad), 
                np.degrees(ns_ra_past_rad), np.degrees(ns_dec_past_rad)
            )

           # prob, dist = self.calculate_overlap_2d(np.degrees(ra_past_rad), np.degrees(dec_past_rad), np.degrees(ns_ra_past_rad), np.degrees(ns_dec_past_rad))

            probs.append(prob)
            dists.append(dist)
            
        df['crossing_probability'] = probs
        df['min_distance_deg'] = dists
        return df

    def run_time_series(self, candidates, eval_times):
        """Runs the exact time-step scanner for top candidates."""
        print(f"\nCalculating time series probability for {len(candidates)} top candidates...")
        
        # Cache the target's trajectory
        pulsar_history_ra, pulsar_history_dec = {}, {}
        for t in eval_times:
            ns_ra_past_rad, ns_dec_past_rad = self.ep.propagate_pos(
                self.ns_ra_arr, self.ns_dec_arr, self.ns_px_arr, 
                self.ns_pmra_samples, self.ns_pmdec_samples, self.ns_rv_samples, 2016.0, 2016.0 - t
            )
            pulsar_history_ra[t] = np.degrees(ns_ra_past_rad)
            pulsar_history_dec[t] = np.degrees(ns_dec_past_rad)

        prob_over_time = {}
        peak_probabilities, peak_ages = [], []

        for idx, (index, row) in enumerate(candidates.iterrows()):
            np.random.seed(index)
            pmra_samples = np.random.normal(loc=row['pmra'], scale=row['pmra_error'], size=self.N)
            pmdec_samples = np.random.normal(loc=row['pmdec'], scale=row['pmdec_error'], size=self.N)
            
            ra_rad_arr = np.full(self.N, np.radians(row['ra']))
            dec_rad_arr = np.full(self.N, np.radians(row['dec']))
            px_arr = np.full(self.N, row['parallax'])
            rv_samples = np.random.normal(loc=row['radial_velocity'], scale=row['radial_velocity_error'], size=self.N)
            
            time_probs = []
            max_prob_for_star, age_at_max = 0.0, 0
            
            for t in eval_times:
                ra_past_rad, dec_past_rad = self.ep.propagate_pos(
                    ra_rad_arr, dec_rad_arr, px_arr, 
                    pmra_samples, pmdec_samples, rv_samples, 2016.0, 2016.0 - t
                )
                ra_past, dec_past = np.degrees(ra_past_rad), np.degrees(dec_past_rad)
                
                ns_ra_past = pulsar_history_ra[t]
                ns_dec_past = pulsar_history_dec[t]

                prob, _ = self.calculate_overlap_2d_ellipse(ra_past, dec_past, ns_ra_past, ns_dec_past)
                # prob, _ = self.calculate_overlap_2d(ra_past, dec_past, ns_ra_past, ns_dec_past)

                time_probs.append(prob)

                if prob > max_prob_for_star:
                    max_prob_for_star = prob
                    age_at_max = t
                    
            prob_over_time[row['source_id']] = time_probs
            peak_probabilities.append(max_prob_for_star)
            peak_ages.append(age_at_max)

        candidates['peak_probability'] = peak_probabilities
        candidates['kinematic_age'] = peak_ages
        
        # Sort and return Top 10
        candidates = candidates.sort_values(by=['peak_probability', 'min_distance_deg'], ascending=[False, True])
        return candidates.head(10), prob_over_time

class Visualizer:
    def __init__(self, target_id, top_candidates):
        """Initializes the visualizer and generates consistent colors for the top candidates."""
        self.target_id = target_id
        self.top_candidates = top_candidates
        
        # Generar colores fijos para las candidatas
        self.colors = cm.rainbow(np.linspace(0, 1, len(self.top_candidates)))
        np.random.seed(99) 
        np.random.shuffle(self.colors)

    def export_excel(self, prob_over_time, eval_times, filename='Top_Candidates_Analysis.xlsx'):
        """Exports the detailed results to an Excel file."""
        print("\nGenerating professional Excel report with two sheets...")

        cols_info = [
            'source_id', 'peak_probability', 'kinematic_age', 'crossing_probability', 'min_distance_deg',
            'ra', 'dec', 'distance_pc', 'v_transverse', 'radial_velocity',
            'abs_mag_g', 'bp_rp_color', 'A_V'
        ]
        df_info = self.top_candidates[cols_info].copy()

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

        prob_data = {'Gaia DR3 Source ID': self.top_candidates['source_id'].values}

        for t_idx, t_val in enumerate(eval_times):
            col_name = f"{int(t_val/1000)} kyr"
            prob_data[col_name] = [prob_over_time[sid][t_idx] for sid in self.top_candidates['source_id']]

        df_probs = pd.DataFrame(prob_data)

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_info.to_excel(writer, sheet_name='Astrophysical_Details', index=False)
            df_probs.to_excel(writer, sheet_name='Time_Series_Probs', index=False)

        print(f"Excel file created: {filename}")

    def plot_time_series(self, prob_over_time, eval_times, tau_mean):
        """Plots the kinematic intersection probability over time."""
        plt.figure(figsize=(10, 6))

        for idx, (index, row) in enumerate(self.top_candidates.iterrows()):
            source = row['source_id']
            probs = prob_over_time[source]
            color = self.colors[idx] 
            
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

    def plot_hr_diagram(self, df, target_star):
        """Plots the Hertzsprung-Russell Diagram for the filtered dataset and top candidates."""
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

        for idx, (_, row) in enumerate(self.top_candidates.iterrows()):
            star_data_hr = df_hr[df_hr['source_id'] == row['source_id']]
            if not star_data_hr.empty:
                col = star_data_hr['bp_rp_color'].values[0]
                m_g = star_data_hr['abs_mag_g'].values[0]
                color = self.colors[idx]
                label_text = f'{idx+1}: {row["source_id"]}'
                
                if row["source_id"] == self.target_id:
                    label_text += ' *'
                    plt.scatter(col, m_g, color='yellow', edgecolor='black', s=300, marker='.', zorder=6, label=label_text)
                else:
                    plt.scatter(col, m_g, color=color, edgecolor='black', s=200, marker='.', zorder=5, label=label_text)
                
                plt.text(col + 0.05, m_g, str(idx + 1), color='black', fontsize=10, fontweight='bold', zorder=7)

        t_col = target_star['bp_rp_color']
        t_mg = target_star['abs_mag_g']
        plt.scatter(t_col, t_mg, color='yellow', edgecolor='black', s=500, marker='.', zorder=10, label=f'Pulsar: {self.target_id}')
        
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
    
    def plot_fits_map(self, fits_file, ep, target_star, tau_mean, tau_error, N, number_sigma):
        """Plots the FITS map with the target and top candidates."""
        print("\nGenerating FITS Map...")
        image = fits.open(fits_file)
        data = image[0].data 
        w = WCS(image[0].header)

        def get_pixel(ra_val, dec_val):
            c = SkyCoord(ra_val * u.deg, dec_val * u.deg, frame='icrs')
            return w.world_to_pixel(SkyCoord(c.galactic.l.value * u.deg, c.galactic.b.value * u.deg, frame='galactic'))

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(1, 1, 1, projection=w)

        vmin_auto = np.nanpercentile(data, 1)
        vmax_auto = np.nanpercentile(data, 99)
        ax1.imshow(data, vmin=vmin_auto, vmax=vmax_auto, cmap='Greys', origin='lower')

        ns_ra, ns_dec = target_star['ra'], target_star['dec']
        ns_parallax, ns_rv = target_star['parallax'], target_star['radial_velocity']
        ns_pmra, ns_pmdec = target_star['pmra'], target_star['pmdec']
        ns_pmra_err, ns_pmdec_err = target_star['pmra_error'], target_star['pmdec_error']

        ns_ra_cen_rad, ns_dec_cen_rad = ep.propagate_pos(
            np.radians(ns_ra), np.radians(ns_dec), ns_parallax, 
            ns_pmra, ns_pmdec, ns_rv, 2016.0, 2016.0 - tau_mean
        )

        target_past_x, target_past_y = get_pixel(np.degrees(ns_ra_cen_rad), np.degrees(ns_dec_cen_rad))
        target_past_x, target_past_y = float(target_past_x), float(target_past_y)

        ax1.plot(target_past_x, target_past_y, '.', color='yellow', markersize=15, markeredgecolor='black', zorder=10)

        target_curr_x, target_curr_y = get_pixel(ns_ra, ns_dec)
        target_curr_x, target_curr_y = float(target_curr_x), float(target_curr_y)
        
        ax1.plot([target_past_x, target_curr_x], [target_past_y, target_curr_y], '-', color='yellow', linewidth=2, zorder=5)
        ax1.plot(target_curr_x, target_curr_y, '.', color='yellow', markersize=14, markeredgecolor='black', label=f'Pulsar: {self.target_id}', zorder=10)

        # Propagamos al púlsar al pasado para tener sus 1000 posiciones
        target_ra_mc_rad, target_dec_mc_rad = ep.propagate_pos(
            np.full(N, np.radians(ns_ra)), np.full(N, np.radians(ns_dec)),
            np.full(N, target_star['parallax']), 
            np.random.normal(ns_pmra, ns_pmra_err, N), 
            np.random.normal(ns_pmdec, ns_pmdec_err, N), 
            np.random.normal(ns_rv, target_star['radial_velocity_error'], N), 2016.0, 2016.0 - TAU_PARAMS['max']
        )
        # Convertimos esas 1000 posiciones pasadas a píxeles de la imagen FITS
        target_px_x_mc, target_px_y_mc = get_pixel(np.degrees(target_ra_mc_rad), np.degrees(target_dec_mc_rad))
        
        # Dibujamos las 1000 líneas completas
        for i in range(N):
            ax1.plot([target_curr_x, target_px_x_mc[i]], [target_curr_y, target_px_y_mc[i]], '-', color='black', alpha=0.02, linewidth=1, zorder=2)

        np.random.seed(888)
        draw_pmra_samples = np.random.normal(loc=ns_pmra, scale=ns_pmra_err, size=N)
        draw_pmdec_samples = np.random.normal(loc=ns_pmdec, scale=ns_pmdec_err, size=N)
        draw_ra_rad_arr = np.full(N, np.radians(ns_ra))
        draw_dec_rad_arr = np.full(N, np.radians(ns_dec))
        draw_px_arr = np.full(N, ns_parallax)
        draw_rv_arr =np.random.normal(ns_rv, target_star['radial_velocity_error'], N)

        draw_ra_past_rad, draw_dec_past_rad = ep.propagate_pos(
            draw_ra_rad_arr, draw_dec_rad_arr, draw_px_arr, 
            draw_pmra_samples, draw_pmdec_samples, draw_rv_arr, 2016.0, 2016.0 - tau_mean
        )

        px_x_mc, px_y_mc = get_pixel(np.degrees(draw_ra_past_rad), np.degrees(draw_dec_past_rad))

        cov_px = np.cov(px_x_mc, px_y_mc)
        evals, evecs = np.linalg.eigh(cov_px)
        width_px = 2 * number_sigma * np.sqrt(evals[1])
        height_px = 2 * number_sigma * np.sqrt(evals[0])
        angle_deg = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))

        # rad = np.sqrt(np.std(px_x_mc)**2 + np.std(px_y_mc)**2) * number_sigma
        # width_px = 2 * rad
        # height_px = 2 * rad
        # angle_deg = 0

        target_ellipse = Ellipse((target_past_x, target_past_y), 
                                 width=width_px, height=height_px, angle=angle_deg,
                                 edgecolor='red', facecolor='none', linestyle='--', 
                                 linewidth=2, alpha=0.8, zorder=4)
        ax1.add_patch(target_ellipse)

        for idx, (index, row) in enumerate(self.top_candidates.iterrows()):
            color = self.colors[idx]
            
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
            rv_samples = np.random.normal(loc=row['radial_velocity'], scale=row['radial_velocity_error'], size=N)

            ra_past_rad, dec_past_rad = ep.propagate_pos(
                ra_rad_arr, dec_rad_arr, px_arr, 
                pmra_samples, pmdec_samples, rv_samples, 2016.0, 2016.0 - tau_samples
            )
            ra_past_samples = np.degrees(ra_past_rad)
            dec_past_samples = np.degrees(dec_past_rad)
            
            X_num, Y_num = get_pixel(ra, dec)
            X_float, Y_float = float(X_num), float(Y_num)
            
            ra_cen_rad, dec_cen_rad = ep.propagate_pos(
                np.radians(ra), np.radians(dec), row['parallax'], 
                pmra, pmdec, row['radial_velocity'], 2016.0, 2016.0 - tau_mean
            )
            x_central, y_central = get_pixel(np.degrees(ra_cen_rad), np.degrees(dec_cen_rad))
            x_central, y_central = float(x_central), float(y_central)

            for i in range(N):
                x_pix, y_pix = get_pixel(ra_past_samples[i], dec_past_samples[i])
                ax1.plot([float(x_pix), X_float], [float(y_pix), Y_float], '-', color=color, alpha=0.03, zorder=1)

            ax1.plot([x_central, X_float], [y_central, Y_float], '-', color=color, linewidth=2, zorder=5)
            ax1.plot(x_central, y_central, 'X', color=color, markersize=8, markeredgecolor='black', zorder=6)
            
            label_text = f'{idx+1}: {row["source_id"]}'
            if row["source_id"] == self.target_id:
                label_text += ' *'
                
            ax1.plot(X_float, Y_float, 'o', color=color, markersize=8, markeredgecolor='white', label=label_text, zorder=7)
            ax1.text(X_float + 0.05, Y_float, str(idx + 1), color='black', fontsize=8, fontweight='bold', zorder=7)
            
        center_ra, center_dec = 312.75, 30.667 
        center_x, center_y = get_pixel(center_ra, center_dec)
        center_x, center_y = float(center_x), float(center_y)

        #ax1.set_xlim([center_x - 100, center_x + 100])
        #ax1.set_ylim([center_y - 100, center_y + 100])

        plt.title('Kinematic Intersection (Moving Target)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=8)
        plt.show(block=False)

    def create_gif(self, df, ep, target_star, tau_min, tau_max, N, number_sigma):
        """Creates the animated GIF of the trajectories over time."""
        output_folder = 'runaway_frames'
        gif_filename = 'runaway_evolution.gif'

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        time_centers = np.arange(tau_min, tau_max, 5000)
        time_window = 10000

        print("\nCreating the GIF...")

        np.random.seed(888)
        ns_ra, ns_dec = target_star['ra'], target_star['dec']
        ns_parallax, ns_rv = target_star['parallax'], target_star['radial_velocity']
        ns_pmra, ns_pmdec = target_star['pmra'], target_star['pmdec']
        ns_pmra_err, ns_pmdec_err = target_star['pmra_error'], target_star['pmdec_error']

        ns_pmra_samples = np.random.normal(loc=ns_pmra, scale=ns_pmra_err, size=N)
        ns_pmdec_samples = np.random.normal(loc=ns_pmdec, scale=ns_pmdec_err, size=N)

        ns_ra_rad_arr = np.full(N, np.radians(ns_ra))
        ns_dec_rad_arr = np.full(N, np.radians(ns_dec))
        ns_px_arr = np.full(N, ns_parallax)
        ns_rv_arr =np.random.normal(ns_rv, target_star['radial_velocity_error'], N)

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
            cov_deg = np.cov(ns_ra_cloud, ns_dec_cloud)
            evals, evecs = np.linalg.eigh(cov_deg)

            width_deg = 2 * number_sigma * np.sqrt(evals[1])
            height_deg = 2 * number_sigma * np.sqrt(evals[0])
            angle_deg = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))

            # rad = np.sqrt(np.std(ns_ra_cloud)**2 + np.std(ns_dec_cloud)**2) * number_sigma
            # width_deg = 2 * rad
            # height_deg = 2 * rad
            # angle_deg = 0

            print(f"     [DEBUG] Elipse: Ancho={width_deg:.6f}°, Alto={height_deg:.6f}°, Ángulo={angle_deg:.2f}°")
            
            ax1.plot([ns_ra, ns_ra_max], [ns_dec, ns_dec_max], '--', color='yellow', alpha=0.4, linewidth=1)
            target_label = f"{self.target_id} *"
            ax1.plot(ns_ra_mean, ns_dec_mean, 'o', color='yellow', markersize=10, markeredgecolor='black', zorder=10, label=target_label)
            ax2.plot(ns_ra_mean, ns_dec_mean, 'o', color='yellow', markersize=10, markeredgecolor='black', zorder=10)
            #ax2.plot(ns_ra_cloud, ns_dec_cloud, '.', color='yellow', alpha=0.05, markersize=3, zorder=6)
            ax2.scatter(ns_ra_cloud, ns_dec_cloud, color='yellow', s=5, alpha=0.5, edgecolor='none', zorder=6)

            ellipse1 = Ellipse((ns_ra_mean, ns_dec_mean), width=width_deg, height=height_deg, angle=angle_deg, 
                               edgecolor='red', facecolor='none', linestyle='--', linewidth=2, zorder=4)
            ellipse2 = Ellipse((ns_ra_mean, ns_dec_mean), width=width_deg, height=height_deg, angle=angle_deg, 
                               edgecolor='red', facecolor='none', linestyle='--', linewidth=2, zorder=4)
            
            ax1.add_patch(ellipse1)
            ax2.add_patch(ellipse2)

            ax1.set_title(f'Exact Trajectories\n Time: {t_kyr:.0f} kyr ago', fontsize=14, fontweight='bold')
            
            t_start = max(0, t_mid - (time_window / 2.0))
            t_end = t_mid + (time_window / 2.0)
            bin_label = f'{t_start/1000:.1f} to {t_end/1000:.1f} kyr ago'
            ax2.set_title(f'Monte Carlo Cloud\n Time: {bin_label}', fontsize=14, fontweight='bold')

            for idx, (index, row) in enumerate(self.top_candidates.iterrows()):
                color = self.colors[idx]
                label_text = f'{idx+1}: {row["source_id"]}'
                
                ra_mid_rad, dec_mid_rad = ep.propagate_pos(
                    np.radians(row['ra']), np.radians(row['dec']), row['parallax'], 
                    row['pmra'], row['pmdec'], row['radial_velocity'], 2016.0, 2016.0 - t_mid
                )
                ra_mid = np.degrees(ra_mid_rad)
                dec_mid = np.degrees(dec_mid_rad)
                
                ax1.plot([row['ra'], ra_mid], [row['dec'], dec_mid], '-', color=color, linewidth=2, alpha=0.5)
                ax1.plot(ra_mid, dec_mid, 'o', color=color, markersize=8, markeredgecolor='black', zorder=9, label=label_text)
                
                np.random.seed(index) 
                pmra_samples = np.random.normal(loc=row['pmra'], scale=row['pmra_error'], size=N)
                pmdec_samples = np.random.normal(loc=row['pmdec'], scale=row['pmdec_error'], size=N)
                
                # Dibujar la nube densa de la candidata 
                ra_rad_arr = np.full(N, np.radians(row['ra']))
                dec_rad_arr = np.full(N, np.radians(row['dec']))
                px_arr = np.full(N, row['parallax'])
                rv_samples = np.random.normal(loc=row['radial_velocity'], scale=row['radial_velocity_error'], size=N)
                
                cand_ra_mc_rad, cand_dec_mc_rad = ep.propagate_pos(
                    ra_rad_arr, dec_rad_arr, px_arr, 
                    pmra_samples, pmdec_samples, rv_samples, 2016.0, 2016.0 - t_mid
                )
                
                ax2.scatter(np.degrees(cand_ra_mc_rad), np.degrees(cand_dec_mc_rad), 
                            color=color, s=5, alpha=0.5, edgecolor='none', zorder=7)
                """
                # Dibujar las lineas de la candidata 
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
                    """
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

if __name__ == "__main__":
    np.random.seed(42)

    # initial config
    CSV_FILE = 'stars2.csv'
    FITS_FILE = 'cygnus.fits'
    #TARGET_ID = '1859461255653506176'
    #TARGET_ID = '1858717401682424064'
    TARGET_ID = '1858675134890513152'
    
    TAU_PARAMS = {'mean': 100280, 'error': 30020, 'min': 70000, 'max': 135000}
    SIM_PARAMS = {'N': 1000, 'sigma': 3}
    
    catalog = StarCatalog(CSV_FILE)
    filtered_stars = catalog.apply_filters()
    target_star = catalog.get_target_star(TARGET_ID)

    filtered_stars = filtered_stars[filtered_stars['source_id'] != str(TARGET_ID)].copy()
    
    simulator = KinematicSimulator(target_star, TAU_PARAMS, SIM_PARAMS)
    
    analyzed_stars = simulator.run_global_scan(filtered_stars)
    candidates = analyzed_stars[analyzed_stars['crossing_probability'] > 0].copy()
    
    if not candidates.empty:
        eval_times = np.arange(TAU_PARAMS['min'], TAU_PARAMS['max'], 1000)
        top_candidates, prob_over_time = simulator.run_time_series(candidates, eval_times)
        
        print("\nTOP CANDIDATES FOUND:")
        print(top_candidates[['source_id', 'peak_probability', 'kinematic_age', 'min_distance_deg']])
        
        viz = Visualizer(TARGET_ID, top_candidates)
        
        viz.export_excel(prob_over_time, eval_times)
        viz.plot_time_series(prob_over_time, eval_times, TAU_PARAMS['mean'])
        viz.plot_hr_diagram(filtered_stars, target_star)
        viz.plot_fits_map(FITS_FILE, simulator.ep, target_star, TAU_PARAMS['mean'], TAU_PARAMS['error'], SIM_PARAMS['N'], SIM_PARAMS['sigma'])
        viz.create_gif(filtered_stars, simulator.ep, target_star, TAU_PARAMS['min'], TAU_PARAMS['max'], SIM_PARAMS['N'], SIM_PARAMS['sigma'])
        
        plt.show()
    else:
        print("\n No candidates found in this run.")