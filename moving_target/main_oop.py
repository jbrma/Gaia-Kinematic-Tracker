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
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

class StarCatalog:
    def __init__(self, csv_file, av_file='Av_Cygnus_Loop.txt'):
        """Initializes the catalog by loading the raw data."""
        self.csv_file = csv_file
        self.av_file = av_file
        
        self.df = pd.read_csv(self.csv_file, dtype={'source_id': str})
        self.df['radial_velocity'] = self.df['radial_velocity'].fillna(0.0)
        self.df['radial_velocity_error'] = self.df['radial_velocity_error'].fillna(17.32)
        
        print(f"Original stars: {len(self.df)}.")

    def apply_filters(self):
        """Applies astrophysical filters, extinction, and kinematics."""
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
        self.ns_px_samples = np.random.normal(loc=self.target['parallax'], scale=self.target['parallax_error'], size=self.N)

        if self.target['radial_velocity_error'] == 17.32 or self.target['radial_velocity_error'] == 0.0:
            self.ns_rv_samples = np.random.normal(0.0, 101.0, self.N)
        else:
            self.ns_rv_samples = np.random.normal(self.target['radial_velocity'], self.target['radial_velocity_error'], self.N)

        #self.ns_rv_samples = np.random.normal(loc=self.target['radial_velocity'], scale=self.target['radial_velocity_error'], size=self.N)
        
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

    def spherical_to_cartesian(self, ra_rad, dec_rad, parallax_mas):
        """Converts spherical coordinates (Gaia) to galactic Cartesian coordinates (parsecs)."""
        plx_safe = np.where(parallax_mas > 0.1, parallax_mas, 0.1)
        d_pc = 1000.0 / plx_safe
        
        x = d_pc * np.cos(dec_rad) * np.cos(ra_rad)
        y = d_pc * np.cos(dec_rad) * np.sin(ra_rad)
        z = d_pc * np.sin(dec_rad)
        
        return x, y, z

    def calculate_overlap_3d_ellipsoid(self, ra_cand, dec_cand, px_cand, ra_target, dec_target, px_target):
        """Calculate the overlap using 3D covariance ellipsoids in parsecs."""
        # Convert both clouds to XYZ (parsecs)
        x_c, y_c, z_c = self.spherical_to_cartesian(ra_cand, dec_cand, px_cand)
        x_t, y_t, z_t = self.spherical_to_cartesian(ra_target, dec_target, px_target)
        
        # Geometric centers
        xc_mean, yc_mean, zc_mean = np.mean(x_c), np.mean(y_c), np.mean(z_c)
        xt_mean, yt_mean, zt_mean = np.mean(x_t), np.mean(y_t), np.mean(z_t)
        
        # Pulsar points within the candidate ellipsoid
        # Calculate the candidate's 3x3 covariance matrix
        cov_c = np.cov(np.vstack((x_c - xc_mean, y_c - yc_mean, z_c - zc_mean)))
        evals_c, evecs_c = np.linalg.eigh(cov_c)
        
        # Extract the 3 semi-axes (scaled by sigma)
        # (We use the maximum value to avoid negligible negative numbers caused by computer rounding)
        axes_c = np.sqrt(np.maximum(evals_c, 1e-10)) * self.n_sigma
        
        # Translate and rotate the Pulsar points to the Candidate's coordinate system
        points_t = np.vstack((x_t - xc_mean, y_t - yc_mean, z_t - zc_mean))
        points_t_rot = np.dot(evecs_c.T, points_t)
        
        # ellipsoid equation 
        frac_target = np.sum((points_t_rot[0]/axes_c[0])**2 + 
                             (points_t_rot[1]/axes_c[1])**2 + 
                             (points_t_rot[2]/axes_c[2])**2 < 1.0) / self.N

        # the candidate's points within the pulsar ellipsoid 
        cov_t = np.cov(np.vstack((x_t - xt_mean, y_t - yt_mean, z_t - zt_mean)))
        evals_t, evecs_t = np.linalg.eigh(cov_t)
        axes_t = np.sqrt(np.maximum(evals_t, 1e-10)) * self.n_sigma
        
        points_c = np.vstack((x_c - xt_mean, y_c - yt_mean, z_c - zt_mean))
        points_c_rot = np.dot(evecs_t.T, points_c)
        
        frac_cand = np.sum((points_c_rot[0]/axes_t[0])**2 + 
                           (points_c_rot[1]/axes_t[1])**2 + 
                           (points_c_rot[2]/axes_t[2])**2 < 1.0) / self.N
        
        final_prob = (frac_target * frac_cand)
        
        # Physical distance between the centers (to see how many parsecs apart they are)
        min_dist_pc = np.sqrt((xc_mean - xt_mean)**2 + (yc_mean - yt_mean)**2 + (zc_mean - zt_mean)**2)
        
        return final_prob, min_dist_pc

    def run_global_scan(self, df):
        """Runs the global probability scan for all stars."""
        print(f"\nStarting global Monte Carlo scan (N={self.N}) for {len(df)} stars...")
        probs, dists = [], []
        
        for i, row in df.iterrows():
            np.random.seed(i)
            pmra_samples = np.random.normal(loc=row['pmra'], scale=row['pmra_error'], size=self.N)
            pmdec_samples = np.random.normal(loc=row['pmdec'], scale=row['pmdec_error'], size=self.N)
            tau_samples = np.random.normal(loc=self.tau_mean, scale=self.tau_error, size=self.N)
            px_samples = np.random.normal(loc=row['parallax'], scale=row['parallax_error'], size=self.N)
            cand_rv_samples = np.random.normal(row['radial_velocity'], row['radial_velocity_error'], self.N)

            ra_rad_arr = np.full(self.N, np.radians(row['ra']))
            dec_rad_arr = np.full(self.N, np.radians(row['dec']))


            ra_past_rad, dec_past_rad = self.ep.propagate_pos(
                ra_rad_arr, dec_rad_arr, px_samples, 
                pmra_samples, pmdec_samples, cand_rv_samples, 2016.0, 2016.0 - tau_samples
            )
            
            ns_ra_past_rad, ns_dec_past_rad = self.ep.propagate_pos(
                self.ns_ra_arr, self.ns_dec_arr, self.ns_px_samples, 
                self.ns_pmra_samples, self.ns_pmdec_samples, self.ns_rv_samples, 2016.0, 2016.0 - tau_samples
            )

            #prob, dist = self.calculate_overlap_2d_ellipse(np.degrees(ra_past_rad), np.degrees(dec_past_rad), np.degrees(ns_ra_past_rad), np.degrees(ns_dec_past_rad))

            prob, dist = self.calculate_overlap_3d_ellipsoid(ra_past_rad, dec_past_rad, px_samples, ns_ra_past_rad, ns_dec_past_rad, self.ns_px_samples)

           # prob, dist = self.calculate_overlap_2d(np.degrees(ra_past_rad), np.degrees(dec_past_rad), np.degrees(ns_ra_past_rad), np.degrees(ns_dec_past_rad))

            probs.append(prob)
            dists.append(dist)
            
        df['crossing_probability'] = probs
        df['min_distance_pc'] = dists
        return df

    def run_time_series(self, candidates, eval_times):
        """Runs the exact time-step scanner for top candidates."""
        print(f"\nCalculating time series probability for {len(candidates)} top candidates...")
        
        # Cache the target's trajectory
        pulsar_history_ra, pulsar_history_dec = {}, {}
        for t in eval_times:
            ns_ra_past_rad, ns_dec_past_rad = self.ep.propagate_pos(
                self.ns_ra_arr, self.ns_dec_arr, self.ns_px_samples, 
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
            px_samples = np.random.normal(loc=row['parallax'], scale=row['parallax_error'], size=self.N)
            rv_samples = np.random.normal(loc=row['radial_velocity'], scale=row['radial_velocity_error'], size=self.N)
            
            time_probs = []
            max_prob_for_star, age_at_max = 0.0, 0
            
            for t in eval_times:
                ra_past_rad, dec_past_rad = self.ep.propagate_pos(
                    ra_rad_arr, dec_rad_arr, px_samples, 
                    pmra_samples, pmdec_samples, rv_samples, 2016.0, 2016.0 - t
                )
                ra_past, dec_past = np.degrees(ra_past_rad), np.degrees(dec_past_rad)
                
                ns_ra_past = pulsar_history_ra[t]
                ns_dec_past = pulsar_history_dec[t]

                #prob, _ = self.calculate_overlap_2d(ra_past, dec_past, ns_ra_past, ns_dec_past)
                #prob, _ = self.calculate_overlap_2d_ellipse(ra_past, dec_past, ns_ra_past, ns_dec_past)
                prob, _ = self.calculate_overlap_3d_ellipsoid(ra_past_rad, dec_past_rad, px_samples, ns_ra_past_rad, ns_dec_past_rad, self.ns_px_samples)

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
        candidates = candidates.sort_values(by=['peak_probability', 'min_distance_pc'], ascending=[False, True])
        return candidates.head(10), prob_over_time

class Visualizer:
    def __init__(self, target_id, top_candidates):
        """Initializes the visualizer and generates consistent colors for the top candidates."""
        self.target_id = target_id
        self.top_candidates = top_candidates
        
        self.colors = cm.rainbow(np.linspace(0, 1, len(self.top_candidates)))
        np.random.seed(99) 
        np.random.shuffle(self.colors)

    def export_excel(self, prob_over_time, eval_times, filename='Top_Candidates_Analysis.xlsx'):
        """Exports the detailed results to an Excel file."""
        print("\nGenerating professional Excel report with two sheets...")

        cols_info = [
            'source_id', 'peak_probability', 'kinematic_age', 'crossing_probability', 'min_distance_pc',
            'ra', 'dec', 'distance_pc', 'v_transverse', 'radial_velocity',
            'abs_mag_g', 'bp_rp_color', 'A_V'
        ]
        df_info = self.top_candidates[cols_info].copy()

        df_info.rename(columns={
            'source_id': 'Gaia DR3 Source ID',
            'peak_probability': 'Peak Prob.',
            'kinematic_age': 'Kinematic age',
            'min_distance_pc': 'Min Dist (pc)',
            'crossing_probability': 'Global Hit Prob.',
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

        #plt.vlines(x=1.0, ymin=-5, ymax=5.0, color='red', linestyle='--', linewidth=1.5, label='Color cut (BP-RP < 1)')
        #plt.hlines(y=5.0, xmin=-1, xmax=1.0, color='green', linestyle='--', linewidth=1.5, label='Mag cut (M_G < 5)')

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
            return w.world_to_pixel(c)

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

        np.random.seed(888)
        draw_pmra_samples = np.random.normal(loc=ns_pmra, scale=ns_pmra_err, size=N)
        draw_pmdec_samples = np.random.normal(loc=ns_pmdec, scale=ns_pmdec_err, size=N)
        draw_px_arr = np.random.normal(ns_parallax, target_star['parallax_error'], N)
        
        if target_star['radial_velocity_error'] == 17.32 or target_star['radial_velocity_error'] == 0.0:
            draw_rv_arr = np.random.normal(0.0, 101.0, N)
        else:
            draw_rv_arr = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)

        draw_ra_rad_arr = np.full(N, np.radians(ns_ra))
        draw_dec_rad_arr = np.full(N, np.radians(ns_dec))

        target_ra_mc_rad, target_dec_mc_rad = ep.propagate_pos(
            draw_ra_rad_arr, draw_dec_rad_arr, draw_px_arr,
            draw_pmra_samples, draw_pmdec_samples, draw_rv_arr, 2016.0, 2016.0 - TAU_PARAMS['max']
        )
        target_px_x_mc, target_px_y_mc = get_pixel(np.degrees(target_ra_mc_rad), np.degrees(target_dec_mc_rad))
        
        for i in range(N):
            ax1.plot([target_curr_x, target_px_x_mc[i]], [target_curr_y, target_px_y_mc[i]], '-', color='yellow', alpha=0.02, linewidth=1, zorder=2)

        """
        # Propagamos al púlsar al pasado para tener sus 1000 posiciones
        target_ra_mc_rad, target_dec_mc_rad = ep.propagate_pos(
            np.full(N, np.radians(ns_ra)), np.full(N, np.radians(ns_dec)),
            np.random.normal(target_star['parallax'], target_star['parallax_error'], N),
            np.random.normal(ns_pmra, ns_pmra_err, N), 
            np.random.normal(ns_pmdec, ns_pmdec_err, N), 
            np.random.normal(ns_rv, target_star['radial_velocity_error'], N), 2016.0, 2016.0 - TAU_PARAMS['max']
        )
        # Convertimos esas 1000 posiciones pasadas a píxeles de la imagen FITS
        target_px_x_mc, target_px_y_mc = get_pixel(np.degrees(target_ra_mc_rad), np.degrees(target_dec_mc_rad))
        
        # Dibujamos las 1000 líneas completas
        for i in range(N):
            ax1.plot([target_curr_x, target_px_x_mc[i]], [target_curr_y, target_px_y_mc[i]], '-', color='yellow', alpha=0.02, linewidth=1, zorder=2)

        np.random.seed(888)
        draw_pmra_samples = np.random.normal(loc=ns_pmra, scale=ns_pmra_err, size=N)
        draw_pmdec_samples = np.random.normal(loc=ns_pmdec, scale=ns_pmdec_err, size=N)
        draw_ra_rad_arr = np.full(N, np.radians(ns_ra))
        draw_dec_rad_arr = np.full(N, np.radians(ns_dec))
        draw_px_arr = np.random.normal(ns_parallax, target_star['parallax_error'], N)

        if pd.isna(target_star['radial_velocity']) or target_star['radial_velocity_error'] == 0.0:
            draw_rv_arr = np.random.normal(0.0, 101.0, N)
        else:
            draw_rv_arr = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)

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
                                 edgecolor='yellow', facecolor='yellow', # Relleno amarillo
                                 linewidth=1, alpha=0.3, zorder=4)
        
        target_ellipse = Ellipse((target_past_x, target_past_y), 
                                 width=width_px, height=height_px, angle=angle_deg,
                                 edgecolor='red', facecolor='none', linestyle='--', 
                                 linewidth=2, alpha=0.8, zorder=4)
        
        ax1.add_patch(target_ellipse)
        
        """

        for idx, (index, row) in enumerate(self.top_candidates.iterrows()):
            color = self.colors[idx]
            
            ra, dec = row['ra'], row['dec']
            pmra, e_pmra = row['pmra'], row['pmra_error']
            pmdec, e_pmdec = row['pmdec'], row['pmdec_error']

            np.random.seed(index)
            pmra_samples = np.random.normal(loc=pmra, scale=e_pmra, size=N)
            pmdec_samples = np.random.normal(loc=pmdec, scale=e_pmdec, size=N)
            px_arr = np.random.normal(row['parallax'], row['parallax_error'], N)
            rv_samples = np.random.normal(loc=row['radial_velocity'], scale=row['radial_velocity_error'], size=N)

            ra_rad_arr = np.full(N, np.radians(ra))
            dec_rad_arr = np.full(N, np.radians(dec))
            
            X_num, Y_num = get_pixel(ra, dec)
            X_float, Y_float = float(X_num), float(Y_num)

            # Propagate star to the past
            ra_past_rad, dec_past_rad = ep.propagate_pos(
                ra_rad_arr, dec_rad_arr, px_arr, 
                pmra_samples, pmdec_samples, rv_samples, 2016.0, 2016.0 - TAU_PARAMS['max']
            )

            ra_past_samples = np.degrees(ra_past_rad)
            dec_past_samples = np.degrees(dec_past_rad)

            for i in range(N):
                x_pix, y_pix = get_pixel(ra_past_samples[i], dec_past_samples[i])
                ax1.plot([float(x_pix), X_float], [float(y_pix), Y_float], '-', color=color, alpha=0.03, zorder=1)

            ra_cen_rad, dec_cen_rad = ep.propagate_pos(
                np.radians(ra), np.radians(dec), row['parallax'], 
                pmra, pmdec, row['radial_velocity'], 2016.0, 2016.0 - tau_mean
            )
            x_central, y_central = get_pixel(np.degrees(ra_cen_rad), np.degrees(dec_cen_rad))
            x_central, y_central = float(x_central), float(y_central)

            """
            np.random.seed(index)
            pmra_samples = np.random.normal(loc=pmra, scale=e_pmra, size=N)
            pmdec_samples = np.random.normal(loc=pmdec, scale=e_pmdec, size=N)
            tau_samples = np.random.normal(loc=tau_mean, scale=tau_error, size=N)

            ra_rad_arr = np.full(N, np.radians(ra))
            dec_rad_arr = np.full(N, np.radians(dec))
            px_arr = np.random.normal(row['parallax'], row['parallax_error'], N)
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
            """

            ax1.plot([x_central, X_float], [y_central, Y_float], '-', color=color, linewidth=2, zorder=5)
            ax1.plot(x_central, y_central, 'X', color=color, markersize=8, markeredgecolor='black', zorder=6)
            
            label_text = f'{idx+1}: {row["source_id"]}'
            if row["source_id"] == self.target_id:
                label_text += ' *'
                
            ax1.plot(X_float, Y_float, 'o', color=color, markersize=8, markeredgecolor='white', label=label_text, zorder=7)
            ax1.text(X_float + 0.05, Y_float, str(idx + 1), color='black', fontsize=8, fontweight='bold', zorder=7)

            # ELLIPSOIDS AT THE EXACT MOMENT OF CROSSING ---
            cruce_age = row['kinematic_age']
            
            if cruce_age > 0:
                
                # Propagate the candidate in the exact year of crossbreeding
                ra_cruce_rad, dec_cruce_rad = ep.propagate_pos(
                    ra_rad_arr, dec_rad_arr, px_arr, 
                    pmra_samples, pmdec_samples, rv_samples, 2016.0, 2016.0 - cruce_age
                )
                c_px_x, c_px_y = get_pixel(np.degrees(ra_cruce_rad), np.degrees(dec_cruce_rad))
                
                cov_c = np.cov(c_px_x, c_px_y)
                evals_c, evecs_c = np.linalg.eigh(cov_c)
                w_c = 2 * number_sigma * np.sqrt(evals_c[1])
                h_c = 2 * number_sigma * np.sqrt(evals_c[0])
                angle_c = np.degrees(np.arctan2(evecs_c[1, 1], evecs_c[0, 1]))
                
                cand_ellipse = Ellipse((np.mean(c_px_x), np.mean(c_px_y)), 
                                       width=w_c, height=h_c, angle=angle_c,
                                       edgecolor=color, facecolor=color, 
                                       linewidth=1, alpha=0.3, zorder=8)
                ax1.add_patch(cand_ellipse)

                # Propagate the pulsar
                p_ra_cruce, p_dec_cruce = ep.propagate_pos(
                    draw_ra_rad_arr, draw_dec_rad_arr, draw_px_arr, 
                    draw_pmra_samples, draw_pmdec_samples, draw_rv_arr, 2016.0, 2016.0 - cruce_age
                )
                p_px_x, p_px_y = get_pixel(np.degrees(p_ra_cruce), np.degrees(p_dec_cruce))
                
                cov_p = np.cov(p_px_x, p_px_y)
                evals_p, evecs_p = np.linalg.eigh(cov_p)
                w_p = 2 * number_sigma * np.sqrt(evals_p[1])
                h_p = 2 * number_sigma * np.sqrt(evals_p[0])
                angle_p = np.degrees(np.arctan2(evecs_p[1, 1], evecs_p[0, 1]))
                
                pulsar_ellipse = Ellipse((np.mean(p_px_x), np.mean(p_px_y)), 
                                         width=w_p, height=h_p, angle=angle_p,
                                         edgecolor=color, facecolor='yellow', 
                                         linewidth=2, alpha=0.4, zorder=8, linestyle='--')
                ax1.add_patch(pulsar_ellipse)

                # Candidate cloud
                ax1.plot(c_px_x, c_px_y, '.', color=color, markersize=1, alpha=0.5, zorder=7)
                
                # Pulsar cloud
                ax1.plot(p_px_x, p_px_y, '.', color='yellow', markersize=1, alpha=0.5, zorder=7)
            
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
        ns_px_samples = np.random.normal(ns_parallax, target_star['parallax_error'], N)
        
        if target_star['radial_velocity_error'] == 17.32 or target_star['radial_velocity_error'] == 0.0:
            ns_rv_arr = np.random.normal(0.0, 101.0, N)
        else:
            ns_rv_arr = np.random.normal(ns_rv, target_star['radial_velocity_error'], N)

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
                ns_ra_rad_arr, ns_dec_rad_arr, ns_px_samples, 
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

            #print(f"     [DEBUG] Elipse: Ancho={width_deg:.6f}°, Alto={height_deg:.6f}°, Ángulo={angle_deg:.2f}°")
            
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
                px_arr = np.random.normal(row['parallax'], row['parallax_error'], N)
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
            
    
    def plot_3d_encounter2D(self, target_star, candidate_row, tau, N=1000):
        """Genera una visualización 3D del encuentro entre el Púlsar y una candidata."""
        from pygaia.astrometry.coordinates import EpochPropagation
        ep = EpochPropagation()
        
        # 1. Propagamos el Púlsar al pasado (tau) con Monte Carlo
        ns_ra_rad, ns_dec_rad = np.radians(target_star['ra']), np.radians(target_star['dec'])
        ns_rv_samples = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)
        
        target_ra_past, target_dec_past = ep.propagate_pos(
            np.full(N, ns_ra_rad), np.full(N, ns_dec_rad), 
            np.random.normal(target_star['parallax'], target_star['parallax_error'], N),
            np.random.normal(target_star['pmra'], target_star['pmra_error'], N),
            np.random.normal(target_star['pmdec'], target_star['pmdec_error'], N),
            ns_rv_samples, 2016.0, 2016.0 - tau
        )
        
        # 2. Propagamos la Candidata al pasado (tau) con Monte Carlo
        c_ra_rad, c_dec_rad = np.radians(candidate_row['ra']), np.radians(candidate_row['dec'])
        c_rv_samples = np.random.normal(candidate_row['radial_velocity'], candidate_row['radial_velocity_error'], N)
        
        cand_ra_past, cand_dec_past = ep.propagate_pos(
            np.full(N, c_ra_rad), np.full(N, c_dec_rad),
            np.random.normal(candidate_row['parallax'], candidate_row['parallax_error'], N),
            np.random.normal(candidate_row['pmra'], candidate_row['pmra_error'], N),
            np.random.normal(candidate_row['pmdec'], candidate_row['pmdec_error'], N),
            c_rv_samples, 2016.0, 2016.0 - tau
        )

        # 3. Convertimos Paralaje (mas) a Distancia (pc) para el eje Z
        # (Usamos la paralaje que devuelve propagate_pos, aunque suele variar poco)
        dist_target = 1000.0 / target_star['parallax'] 
        dist_cand = 1000.0 / candidate_row['parallax']

        # 4. Crear el gráfico 3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        plt.style.use('dark_background')

        # Dibujar nubes
        ax.scatter(np.degrees(target_ra_past), np.degrees(target_dec_past), np.full(N, dist_target), 
                   c='yellow', s=2, alpha=0.3, label=f'Pulsar ({self.target_id})')
        
        ax.scatter(np.degrees(cand_ra_past), np.degrees(cand_dec_past), np.full(N, dist_cand), 
                   c='orange', s=2, alpha=0.3, label=f"Candidate {candidate_row['source_id']}")

        # Configuración estética
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')
        ax.set_zlabel('Distance (pc)')
        ax.set_title(f'Cross point in 3D {tau:,.0f} years ago', fontsize=15)
        ax.legend()
        
        # Guardar y mostrar
        plt.savefig(f"Encuentro_3D_{candidate_row['source_id']}.png", dpi=300)
        plt.show()

    def plot_3d_encounter(self, target_star, candidate_row, tau, N=1000):
        """Genera una visualización 3D en Coordenadas Cartesianas (Parsecs)."""
        from pygaia.astrometry.coordinates import EpochPropagation
        ep = EpochPropagation()
        
        # --- 1. Propagar PÚLSAR al pasado ---
        ns_ra_rad, ns_dec_rad = np.radians(target_star['ra']), np.radians(target_star['dec'])
        ns_rv_samples = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)
        ns_px_samples = np.random.normal(target_star['parallax'], target_star['parallax_error'], N)
        
        target_ra_past, target_dec_past = ep.propagate_pos(
            np.full(N, ns_ra_rad), np.full(N, ns_dec_rad), ns_px_samples,
            np.random.normal(target_star['pmra'], target_star['pmra_error'], N),
            np.random.normal(target_star['pmdec'], target_star['pmdec_error'], N),
            ns_rv_samples, 2016.0, 2016.0 - tau
        )
        
        # --- 2. Propagar CANDIDATA al pasado ---
        c_ra_rad, c_dec_rad = np.radians(candidate_row['ra']), np.radians(candidate_row['dec'])
        c_rv_samples = np.random.normal(candidate_row['radial_velocity'], candidate_row['radial_velocity_error'], N)
        c_px_samples = np.random.normal(candidate_row['parallax'], candidate_row['parallax_error'], N)
        
        cand_ra_past, cand_dec_past = ep.propagate_pos(
            np.full(N, c_ra_rad), np.full(N, c_dec_rad), c_px_samples,
            np.random.normal(candidate_row['pmra'], candidate_row['pmra_error'], N),
            np.random.normal(candidate_row['pmdec'], candidate_row['pmdec_error'], N),
            c_rv_samples, 2016.0, 2016.0 - tau
        )

        # --- 3. CONVERTIR A CARTESIANAS (Parsecs) ---
        # (Usamos la misma lógica que en tu motor 3D)
        def esferico_a_cartesiano(ra, dec, px):
            plx_safe = np.where(px > 0.1, px, 0.1)
            d = 1000.0 / plx_safe
            x = d * np.cos(dec) * np.cos(ra)
            y = d * np.cos(dec) * np.sin(ra)
            z = d * np.sin(dec)
            return x, y, z

        xt, yt, zt = esferico_a_cartesiano(target_ra_past, target_dec_past, ns_px_samples)
        xc, yc, zc = esferico_a_cartesiano(cand_ra_past, cand_dec_past, c_px_samples)

        # --- 4. DIBUJAR EN 3D ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.style.use('dark_background')

        # Nubes de puntos (Los elipsoides de Monte Carlo)
        ax.scatter(xt, yt, zt, c='yellow', s=3, alpha=0.3, label=f'Púlsar ({self.target_id})')
        ax.scatter(xc, yc, zc, c='orange', s=3, alpha=0.3, label=f"Candidata {candidate_row['source_id']}")

        # Pintar los centros (Para ver la distancia geométrica)
        ax.scatter(np.mean(xt), np.mean(yt), np.mean(zt), c='white', s=50, marker='x')
        ax.scatter(np.mean(xc), np.mean(yc), np.mean(zc), c='white', s=50, marker='x')

        # Forzar que los ejes tengan la misma escala (para que los huevos no se deformen visualmente)
        # Encontramos los límites máximos y mínimos de ambas nubes
        max_range = np.array([xt.max()-xt.min(), yt.max()-yt.min(), zt.max()-zt.min(), 
                              xc.max()-xc.min(), yc.max()-yc.min(), zc.max()-zc.min()]).max() / 2.0
        
        mid_x = (np.mean(xt) + np.mean(xc)) / 2.0
        mid_y = (np.mean(yt) + np.mean(yc)) / 2.0
        mid_z = (np.mean(zt) + np.mean(zc)) / 2.0
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Etiquetas
        ax.set_xlabel('X (Parsecs)', color='gray')
        ax.set_ylabel('Y (Parsecs)', color='gray')
        ax.set_zlabel('Z (Parsecs)', color='gray')
        ax.set_title(f'Intersección de Elipsoides hace {tau:,.0f} años\n(Coordenadas Galácticas Cartesianas)', pad=20)
        ax.legend()
        
        plt.show()

    def plot_3d_interactive_plotly(self, target_star, candidates_df, tau, N=1000):
        """Genera un entorno 3D navegable de alto rendimiento en el navegador."""
        from pygaia.astrometry.coordinates import EpochPropagation
        ep = EpochPropagation()
        
        def esferico_a_cartesiano(ra, dec, px):
            plx_safe = np.where(px > 0.1, px, 0.1)
            d = 1000.0 / plx_safe
            x = d * np.cos(dec) * np.cos(ra)
            y = d * np.cos(dec) * np.sin(ra)
            z = d * np.sin(dec)
            return x, y, z

        fig = go.Figure()

        # --- 1. PÚLSAR ---
        ns_ra_rad, ns_dec_rad = np.radians(target_star['ra']), np.radians(target_star['dec'])
        ns_rv_samples = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)
        ns_px_samples = np.random.normal(target_star['parallax'], target_star['parallax_error'], N)
        
        target_ra_past, target_dec_past = ep.propagate_pos(
            np.full(N, ns_ra_rad), np.full(N, ns_dec_rad), ns_px_samples,
            np.random.normal(target_star['pmra'], target_star['pmra_error'], N),
            np.random.normal(target_star['pmdec'], target_star['pmdec_error'], N),
            ns_rv_samples, 2016.0, 2016.0 - tau
        )
        
        xt, yt, zt = esferico_a_cartesiano(target_ra_past, target_dec_past, ns_px_samples)
        
        # Nube del Púlsar
        fig.add_trace(go.Scatter3d(
            x=xt, y=yt, z=zt, mode='markers',
            marker=dict(size=2, color='yellow', opacity=0.3),
            name=f'Púlsar ({self.target_id})'
        ))
        # Centro del Púlsar
        fig.add_trace(go.Scatter3d(
            x=[np.mean(xt)], y=[np.mean(yt)], z=[np.mean(zt)], mode='markers',
            marker=dict(size=6, color='white', symbol='cross'),
            name='Centro Púlsar'
        ))

        # --- 2. CANDIDATAS ---
        for idx, (_, row) in enumerate(candidates_df.iterrows()):
            c_ra_rad, c_dec_rad = np.radians(row['ra']), np.radians(row['dec'])
            c_rv_samples = np.random.normal(row['radial_velocity'], row['radial_velocity_error'], N)
            c_px_samples = np.random.normal(row['parallax'], row['parallax_error'], N)
            
            cand_ra_past, cand_dec_past = ep.propagate_pos(
                np.full(N, c_ra_rad), np.full(N, c_dec_rad), c_px_samples,
                np.random.normal(row['pmra'], row['pmra_error'], N),
                np.random.normal(row['pmdec'], row['pmdec_error'], N),
                c_rv_samples, 2016.0, 2016.0 - tau
            )

            xc, yc, zc = esferico_a_cartesiano(cand_ra_past, cand_dec_past, c_px_samples)
            
            # Nube Candidata (Plotly asigna colores automáticamente)
            fig.add_trace(go.Scatter3d(
                x=xc, y=yc, z=zc, mode='markers',
                marker=dict(size=2, opacity=0.2),
                name=f"Cand {row['source_id']}"
            ))

        # --- 3. CONFIGURACIÓN DEL ENTORNO 3D ---
        fig.update_layout(
            title=f'Simulación Cinemática Interactiva (Hace {tau:,.0f} años)',
            scene=dict(
                xaxis_title='X (Parsecs)',
                yaxis_title='Y (Parsecs)',
                zaxis_title='Z (Parsecs)',
                aspectmode='data', # ¡CLAVE! Esto fuerza la escala 1:1 real
                bgcolor='black'
            ),
            paper_bgcolor='black',
            font=dict(color='white'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # Esto abrirá una pestaña nueva en tu navegador web
        fig.show()

    def plot_3d_trajectories_plotly(self, target_star, candidates_df, tau, N=1000):
        """Muestra las trayectorias como estelas de humo (nubes de probabilidad a lo largo del tiempo)."""
        import plotly.graph_objects as go
        from pygaia.astrometry.coordinates import EpochPropagation
        ep = EpochPropagation()

        def esferico_a_cartesiano(ra, dec, px):
            plx_safe = np.where(px > 0.1, px, 0.1)
            d = 1000.0 / plx_safe
            x = d * np.cos(dec) * np.cos(ra)
            y = d * np.cos(dec) * np.sin(ra)
            z = d * np.sin(dec)
            return x, y, z

        fig = go.Figure()
        
        # Hacemos 15 "fotos" a lo largo de los tau años
        time_steps = np.linspace(0, tau, 15) 
        
        # Para calcular los límites visuales después
        all_x, all_y, all_z = [], [], []

        # --- 1. ESTELA DEL PÚLSAR (Amarillo) ---
        ns_ra_rad, ns_dec_rad = np.radians(target_star['ra']), np.radians(target_star['dec'])
        ns_rv = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)
        ns_px = np.random.normal(target_star['parallax'], target_star['parallax_error'], N)
        ns_pmra = np.random.normal(target_star['pmra'], target_star['pmra_error'], N)
        ns_pmdec = np.random.normal(target_star['pmdec'], target_star['pmdec_error'], N)

        p_x, p_y, p_z = [], [], []
        # Propagamos la nube entera en cada paso de tiempo
        for t in time_steps:
            ra_t, dec_t = ep.propagate_pos(
                np.full(N, ns_ra_rad), np.full(N, ns_dec_rad), ns_px,
                ns_pmra, ns_pmdec, ns_rv, 2016.0, 2016.0 - t
            )
            x, y, z = esferico_a_cartesiano(ra_t, dec_t, ns_px)
            p_x.extend(x); p_y.extend(y); p_z.extend(z)
        
        all_x.extend(p_x); all_y.extend(p_y); all_z.extend(p_z)

        # Añadimos la estela entera de una sola vez (como puntos translúcidos)
        fig.add_trace(go.Scatter3d(
            x=p_x, y=p_y, z=p_z, mode='markers',
            marker=dict(size=2, color='yellow', opacity=0.03), # <- ¡Baja opacidad clave!
            name=f'Púlsar ({self.target_id})'
        ))

        # --- 2. ESTELAS DE LAS CANDIDATAS ---
        colors = ['cyan', 'magenta', 'lime', 'orange', 'red']
        for idx, (_, row) in enumerate(candidates_df.iterrows()):
            color = colors[idx % len(colors)]
            
            c_ra_rad, c_dec_rad = np.radians(row['ra']), np.radians(row['dec'])
            c_rv = np.random.normal(row['radial_velocity'], row['radial_velocity_error'], N)
            c_px = np.random.normal(row['parallax'], row['parallax_error'], N)
            c_pmra = np.random.normal(row['pmra'], row['pmra_error'], N)
            c_pmdec = np.random.normal(row['pmdec'], row['pmdec_error'], N)
            
            c_x, c_y, c_z = [], [], []
            for t in time_steps:
                ra_t, dec_t = ep.propagate_pos(
                    np.full(N, c_ra_rad), np.full(N, c_dec_rad), c_px,
                    c_pmra, c_pmdec, c_rv, 2016.0, 2016.0 - t
                )
                x, y, z = esferico_a_cartesiano(ra_t, dec_t, c_px)
                c_x.extend(x); c_y.extend(y); c_z.extend(z)
                
            all_x.extend(c_x); all_y.extend(c_y); all_z.extend(c_z)

            fig.add_trace(go.Scatter3d(
                x=c_x, y=c_y, z=c_z, mode='markers',
                marker=dict(size=2, color=color, opacity=0.03),
                name=f"Cand {row['source_id']}"
            ))

        # --- 3. CONFIGURACIÓN DEL ESPACIO ---
        margin = 15 # Parsecs de margen alrededor del grupo
        fig.update_layout(
            title=f"Estelas Cinemáticas: Hoy -> Hace {tau:,.0f} años",
            scene=dict(
                xaxis=dict(title='X (pc)', range=[min(all_x)-margin, max(all_x)+margin]),
                yaxis=dict(title='Y (pc)', range=[min(all_y)-margin, max(all_y)+margin]),
                zaxis=dict(title='Z (pc)', range=[min(all_z)-margin, max(all_z)+margin]),
                aspectmode='data', 
                bgcolor='black'
            ),
            paper_bgcolor='black',
            font=dict(color='white'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show()

    def plot_3d_kinematic_slider(self, target_star, candidates_df, max_tau, steps=50, N=500):
        """
        Generates an interactive 3D slider to visualize the expansion of the 
        uncertainty cloud (Monte Carlo clones) as we move back in time.
        Uses highly contrasted distinct colors for each candidate.
        """
        import plotly.graph_objects as go
        from pygaia.astrometry.coordinates import EpochPropagation
        import numpy as np
        
        ep = EpochPropagation()

        def to_cartesian(ra, dec, px):
            # Clip parallax to avoid division by zero
            safe_px = np.where(px > 0.1, px, 0.1)
            dist = 1000.0 / safe_px
            x = dist * np.cos(dec) * np.cos(ra)
            y = dist * np.cos(dec) * np.sin(ra)
            z = dist * np.sin(dec)
            return x, y, z

        # Time vector (from today to the past)
        time_steps = np.linspace(0, max_tau, steps)
        
        # --- MONTE CARLO SAMPLES (Fixed to prevent "shaking" effect) ---
        # Pulsar Samples
        p_ra, p_dec = np.radians(target_star['ra']), np.radians(target_star['dec'])
        
        if pd.isna(target_star['radial_velocity']) or target_star['radial_velocity_error'] == 0.0:
            p_rv = np.random.normal(0.0, 101.0, N)
        else:
            p_rv = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)

        #p_rv = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)
        p_px = np.random.normal(target_star['parallax'], target_star['parallax_error'], N)
        p_pmra = np.random.normal(target_star['pmra'], target_star['pmra_error'], N)
        p_pmdec = np.random.normal(target_star['pmdec'], target_star['pmdec_error'], N)
        
        # Candidates Samples
        cand_dict = {}
        for _, row in candidates_df.iterrows():
            cand_dict[row['source_id']] = {
                'ra': np.radians(row['ra']), 'dec': np.radians(row['dec']),
                'rv': np.random.normal(row['radial_velocity'], row['radial_velocity_error'], N),
                'px': np.random.normal(row['parallax'], row['parallax_error'], N),
                'pmra': np.random.normal(row['pmra'], row['pmra_error'], N),
                'pmdec': np.random.normal(row['pmdec'], row['pmdec_error'], N),
            }

        print("Computing kinematics for 3D slider...")

        # --- PRE-COMPUTE POSITIONS ---
        frames_data = []
        bound_x, bound_y, bound_z = [], [], []

        for t in time_steps:
            frame = {'pulsar': None, 'cands': {}}
            
            # Pulsar Propagation
            ra_p, dec_p = ep.propagate_pos(
                np.full(N, p_ra), np.full(N, p_dec), p_px,
                p_pmra, p_pmdec, p_rv, 2016.0, 2016.0 - t
            )
            x_p, y_p, z_p = to_cartesian(ra_p, dec_p, p_px)
            frame['pulsar'] = (x_p, y_p, z_p)
            bound_x.extend(x_p); bound_y.extend(y_p); bound_z.extend(z_p)
            
            # Candidates Propagation
            for cid, s in cand_dict.items():
                ra_c, dec_c = ep.propagate_pos(
                    np.full(N, s['ra']), np.full(N, s['dec']), s['px'],
                    s['pmra'], s['pmdec'], s['rv'], 2016.0, 2016.0 - t
                )
                x_c, y_c, z_c = to_cartesian(ra_c, dec_c, s['px'])
                frame['cands'][cid] = (x_c, y_c, z_c)
                bound_x.extend(x_c); bound_y.extend(y_c); bound_z.extend(z_c)
            
            frames_data.append(frame)

        # --- BUILD FIGURE ---
        fig = go.Figure()

        # Initial Trace: Pulsar (Yellow)
        fig.add_trace(go.Scatter3d(
            x=frames_data[0]['pulsar'][0], y=frames_data[0]['pulsar'][1], z=frames_data[0]['pulsar'][2],
            mode='markers', marker=dict(size=2, color='yellow', opacity=0.6),
            name=f'Pulsar {self.target_id}'
        ))

        # Initial Trace: Candidates (Original highly distinct colors)
        candidate_colors = ['cyan', 'magenta', 'lime', 'orange', 'red']
        
        for i, (cid, _) in enumerate(cand_dict.items()):
            fig.add_trace(go.Scatter3d(
                x=frames_data[0]['cands'][cid][0], 
                y=frames_data[0]['cands'][cid][1], 
                z=frames_data[0]['cands'][cid][2],
                mode='markers', marker=dict(size=2, color=candidate_colors[i % len(candidate_colors)], opacity=0.4),
                name=f'Candidate {cid}'
            ))

        # Animation Frames
        fig.frames = [go.Frame(
            data=[go.Scatter3d(x=f['pulsar'][0], y=f['pulsar'][1], z=f['pulsar'][2])] + 
                 [go.Scatter3d(x=f['cands'][cid][0], y=f['cands'][cid][1], z=f['cands'][cid][2]) 
                  for cid in cand_dict.keys()],
            name=str(int(t))
        ) for f, t in zip(frames_data, time_steps)]

        # Layout & Slider Config
        fig.update_layout(
            title="3D Kinematic Expansion & Collision Search",
            template="plotly_dark",
            scene=dict(
                xaxis=dict(title='X (pc)', range=[min(bound_x)-5, max(bound_x)+5]),
                yaxis=dict(title='Y (pc)', range=[min(bound_y)-5, max(bound_y)+5]),
                zaxis=dict(title='Z (pc)', range=[min(bound_z)-5, max(bound_z)+5]),
                aspectmode='data' # Maintains physical 1:1 scale
            ),
            sliders=[{
                "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                           "label": f.name, "method": "animate"} for f in fig.frames],
                "currentvalue": {"prefix": "Time back (years): ", "font": {"size": 18, "color": "white"}}
            }]
        )

        fig.show()

    def plot_3d_ellipsoids_snapshot(self, target_star, candidates_df, tau, N=1000, n_sigma=2.0):
        """
        Generates a static 3D snapshot showing the solid mathematical ellipsoids 
        (covariance meshes) at a specific point in time (tau).
        """
        import plotly.graph_objects as go
        from pygaia.astrometry.coordinates import EpochPropagation
        import numpy as np
        
        ep = EpochPropagation()

        def to_cartesian(ra, dec, px):
            safe_px = np.where(px > 0.1, px, 0.1)
            dist = 1000.0 / safe_px
            x = dist * np.cos(dec) * np.cos(ra)
            y = dist * np.cos(dec) * np.sin(ra)
            z = dist * np.sin(dec)
            return x, y, z

        def get_ellipsoid_mesh(x, y, z, n_sigma=2.0, resolution=30):
            """Calculates the 3D surface mesh of the covariance ellipsoid."""
            # 1. Covariance matrix and center
            cov = np.cov(np.vstack((x, y, z)))
            mean = np.array([np.mean(x), np.mean(y), np.mean(z)])
            
            # 2. Eigenvalues and Eigenvectors
            evals, evecs = np.linalg.eigh(cov)
            radii = np.sqrt(np.maximum(evals, 1e-10)) * n_sigma
            
            # 3. Parametric equations for a unit sphere
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            u, v = np.meshgrid(u, v)
            
            # Scale to ellipsoid radii
            xs = np.cos(u) * np.sin(v) * radii[0]
            ys = np.sin(u) * np.sin(v) * radii[1]
            zs = np.cos(v) * radii[2]
            
            # 4. Rotate by eigenvectors
            points = np.vstack((xs.flatten(), ys.flatten(), zs.flatten()))
            rotated = np.dot(evecs, points)
            
            # 5. Translate to center and reshape for Plotly Surface
            X = rotated[0, :].reshape(resolution, resolution) + mean[0]
            Y = rotated[1, :].reshape(resolution, resolution) + mean[1]
            Z = rotated[2, :].reshape(resolution, resolution) + mean[2]
            
            return X, Y, Z

        print(f"Generating 3D Ellipsoid meshes for tau = {tau} years...")

        fig = go.Figure()
        bound_x, bound_y, bound_z = [], [], []

        # --- 1. PULSAR ELLIPSOID ---
        p_ra, p_dec = np.radians(target_star['ra']), np.radians(target_star['dec'])

        if pd.isna(target_star['radial_velocity']) or target_star['radial_velocity_error'] == 0.0:
            p_rv = np.random.normal(0.0, 101.0, N)
        else:
            p_rv = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)

        #p_rv = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)
        p_px = np.random.normal(target_star['parallax'], target_star['parallax_error'], N)
        p_pmra = np.random.normal(target_star['pmra'], target_star['pmra_error'], N)
        p_pmdec = np.random.normal(target_star['pmdec'], target_star['pmdec_error'], N)
        
        ra_p, dec_p = ep.propagate_pos(np.full(N, p_ra), np.full(N, p_dec), p_px, p_pmra, p_pmdec, p_rv, 2016.0, 2016.0 - tau)
        x_p, y_p, z_p = to_cartesian(ra_p, dec_p, p_px)
        bound_x.extend(x_p); bound_y.extend(y_p); bound_z.extend(z_p)

        # Generate mesh for Pulsar
        X_p, Y_p, Z_p = get_ellipsoid_mesh(x_p, y_p, z_p, n_sigma)
        
        # Add translucent surface
        fig.add_trace(go.Surface(
            x=X_p, y=Y_p, z=Z_p,
            colorscale=[[0, 'yellow'], [1, 'yellow']], # Solid color
            opacity=0.3, showscale=False, name='Pulsar Ellipsoid'
        ))
        # Add points inside
        fig.add_trace(go.Scatter3d(
            x=x_p, y=y_p, z=z_p, mode='markers',
            marker=dict(size=1, color='yellow', opacity=0.2), showlegend=False
        ))

        # --- 2. CANDIDATES ELLIPSOIDS ---
        candidate_colors = ['cyan', 'magenta', 'lime', 'orange', 'red']
        
        for i, (_, row) in enumerate(candidates_df.iterrows()):
            c_color = candidate_colors[i % len(candidate_colors)]
            c_ra, c_dec = np.radians(row['ra']), np.radians(row['dec'])
            c_rv = np.random.normal(row['radial_velocity'], row['radial_velocity_error'], N)
            c_px = np.random.normal(row['parallax'], row['parallax_error'], N)
            c_pmra = np.random.normal(row['pmra'], row['pmra_error'], N)
            c_pmdec = np.random.normal(row['pmdec'], row['pmdec_error'], N)
            
            ra_c, dec_c = ep.propagate_pos(np.full(N, c_ra), np.full(N, c_dec), c_px, c_pmra, c_pmdec, c_rv, 2016.0, 2016.0 - tau)
            x_c, y_c, z_c = to_cartesian(ra_c, dec_c, c_px)
            bound_x.extend(x_c); bound_y.extend(y_c); bound_z.extend(z_c)

            # Generate mesh for Candidate
            X_c, Y_c, Z_c = get_ellipsoid_mesh(x_c, y_c, z_c, n_sigma)
            
            fig.add_trace(go.Surface(
                x=X_c, y=Y_c, z=Z_c,
                colorscale=[[0, c_color], [1, c_color]], 
                opacity=0.3, showscale=False, name=f"Candidate {row['source_id']}"
            ))
            fig.add_trace(go.Scatter3d(
                x=x_c, y=y_c, z=z_c, mode='markers',
                marker=dict(size=1, color=c_color, opacity=0.2), showlegend=False
            ))

        margin = 0.5 
        
        fig.update_layout(
            title=f"Mathematical Intersection of Uncertainty Ellipsoids ({n_sigma} Sigma)<br>Time: -{tau:,.0f} years",
            template="plotly_dark",
            scene=dict(
                xaxis=dict(title='X (pc)', range=[min(bound_x)-margin, max(bound_x)+margin]),
                yaxis=dict(title='Y (pc)', range=[min(bound_y)-margin, max(bound_y)+margin]),
                zaxis=dict(title='Z (pc)', range=[min(bound_z)-margin, max(bound_z)+margin]),
                
                # 2. EL TRUCO DEL ZOOM: 
                # 'data' mantiene la proporción física real (1 parsec en X = 1 parsec en Z).
                # Si aún así te cuesta navegar, cambia 'data' por 'auto'. Deformará los huevos un poco, 
                # pero el zoom y la rotación serán infinitamente más fáciles de controlar.
                aspectmode='data' 
            ),
            margin=dict(l=0, r=0, b=0, t=40) # Esto quita los márgenes blancos de la ventana de Chrome
        )
        fig.show()


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
    #print(len(filtered_stars))
    filtered_stars = filtered_stars[filtered_stars['source_id'] != str(TARGET_ID)].copy()
    
    simulator = KinematicSimulator(target_star, TAU_PARAMS, SIM_PARAMS)
    
    analyzed_stars = simulator.run_global_scan(filtered_stars)
    candidates = analyzed_stars[analyzed_stars['crossing_probability'] > 0].copy()
    
    if not candidates.empty:
        eval_times = np.arange(TAU_PARAMS['min'], TAU_PARAMS['max'], 1000)
        top_candidates, prob_over_time = simulator.run_time_series(candidates, eval_times)
        
        print("\nTOP CANDIDATES FOUND:")
        print(top_candidates[['source_id', 'peak_probability', 'kinematic_age', 'min_distance_pc']])
        
        viz = Visualizer(TARGET_ID, top_candidates)

        #best_cand = top_candidates.iloc[0]
        #viz.plot_3d_encounter(target_star, best_cand, best_cand['kinematic_age'])

        #all
        #viz.plot_3d_interactive_plotly(target_star, top_candidates, TAU_PARAMS['mean'])
        #viz.plot_3d_trajectories_plotly(target_star, top_candidates, TAU_PARAMS['mean'])
        viz.plot_3d_kinematic_slider(target_star, top_candidates, TAU_PARAMS['mean'])

        # Sacamos los datos de la mejor candidata (la fila 0)
        best_cand = top_candidates.iloc[0]
        best_age = best_cand['kinematic_age']
        
        # Le decimos que dibuje los elipsoides matemáticos exactos en ese año
        #viz.plot_3d_ellipsoids_snapshot(target_star, top_candidates, best_age, n_sigma=2.0)
        
        viz.export_excel(prob_over_time, eval_times)
        viz.plot_time_series(prob_over_time, eval_times, TAU_PARAMS['mean'])
        viz.plot_hr_diagram(filtered_stars, target_star)
        viz.plot_fits_map(FITS_FILE, simulator.ep, target_star, TAU_PARAMS['mean'], TAU_PARAMS['error'], SIM_PARAMS['N'], SIM_PARAMS['sigma'])
        viz.create_gif(filtered_stars, simulator.ep, target_star, TAU_PARAMS['min'], TAU_PARAMS['max'], SIM_PARAMS['N'], SIM_PARAMS['sigma'])
        
        plt.show()
    else:
        print("\n No candidates found in this run.")