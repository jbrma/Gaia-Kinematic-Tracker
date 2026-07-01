import os
import shutil
import sys
import time
from pathlib import Path

import astropy.io.fits as fits
import astropy.units as u
import extinction
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.patches import Ellipse
from PIL import Image
from pygaia.astrometry.coordinates import EpochPropagation

class StarCatalog:
    def __init__(self, csv_file, av_file):
        """Initializes the catalog by loading the raw data."""
        self.csv_file = csv_file
        self.av_file = av_file
        
        self.df = pd.read_csv(self.csv_file, dtype={'source_id': str})
        self.df['radial_velocity'] = self.df['radial_velocity'].fillna(0.0)
        self.df['radial_velocity_error'] = self.df['radial_velocity_error'].fillna(17.32)
        
        print(f"Original stars: {len(self.df)}.")

    def apply_filters(self, filter):
        """Applies astrophysical filters, extinction, and kinematics."""
        self.df = self.df.dropna(subset=['parallax', 'pmra', 'pmdec'])

        # distance in parsecs and kiloparsecs
        self.df['distance_pc'] = 1000.0 / self.df['parallax']
        self.df['distance_kpc'] = self.df['distance_pc'] / 1000.0

        self.df['parallax_snr'] = self.df['parallax'] / self.df['parallax_error']
        self.df = self.df[self.df['parallax_snr'] > 5.0]

        # Kinematics basics
        self.df['pm_total'] = np.sqrt(self.df['pmra']**2 + self.df['pmdec']**2)
        self.df['v_transverse'] = 4.74 * (self.df['pm_total'] / self.df['parallax'])

        av_data = np.loadtxt(self.av_file)
        if av_data.ndim == 0:
            av_data = np.array([[0, 0, 0]])
        elif av_data.ndim == 1:
            av_data = np.array([av_data])

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
        if filter:
            self.df = self.df[(self.df['abs_mag_g'] < 5) & (self.df['bp_rp_color'] < 1)]
        
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
    
    def generate_multivariate_clones(self, row, custom_N=None):
        """
        Generate N astrometric clones, taking into account the 
        actual dependencies among the 5 parameters of the Gaia DR3 covariance matrix.
        """
        if custom_N is None:
            custom_N = self.N

        mean_vector = [0.0, 0.0, row['parallax'], row['pmra'], row['pmdec']]

        errors = np.array([
            row['ra_error'], row['dec_error'], 
            row['parallax_error'], row['pmra_error'], row['pmdec_error']
        ])

        def get_corr(col):
            return row[col] if col in row else 0.0

        # Gaias correlation (5x5)
        C = np.array([
            [1.0, get_corr('ra_dec_corr'), get_corr('ra_parallax_corr'), get_corr('ra_pmra_corr'), get_corr('ra_pmdec_corr')],
            [get_corr('ra_dec_corr'), 1.0, get_corr('dec_parallax_corr'), get_corr('dec_pmra_corr'), get_corr('dec_pmdec_corr')],
            [get_corr('ra_parallax_corr'), get_corr('dec_parallax_corr'), 1.0, get_corr('parallax_pmra_corr'), get_corr('parallax_pmdec_corr')],
            [get_corr('ra_pmra_corr'), get_corr('dec_pmra_corr'), get_corr('parallax_pmra_corr'), 1.0, get_corr('pmra_pmdec_corr')],
            [get_corr('ra_pmdec_corr'), get_corr('dec_pmdec_corr'), get_corr('parallax_pmdec_corr'), get_corr('pmra_pmdec_corr'), 1.0]
        ])

        # Build cov matrix
        cov_matrix = np.outer(errors, errors) * C
        samples = np.random.multivariate_normal(mean_vector, cov_matrix, size=custom_N)

        delta_ra_star_mas = samples[:, 0] 
        delta_dec_mas = samples[:, 1]
        px_samples = samples[:, 2]
        pmra_samples = samples[:, 3]
        pmdec_samples = samples[:, 4]

        # Convert to deg
        cos_dec = np.cos(np.radians(row['dec']))
        ra_deg_arr = row['ra'] + (delta_ra_star_mas / cos_dec) / 3600000.0
        dec_deg_arr = row['dec'] + (delta_dec_mas / 3600000.0)

        return np.radians(ra_deg_arr), np.radians(dec_deg_arr), px_samples, pmra_samples, pmdec_samples

    def _init_target_cloud(self):
        """Pre-calculates the static Monte Carlo base cloud for the target star."""
        np.random.seed(888) # Fixed seed for the pulsar

        self.ns_ra_arr, self.ns_dec_arr, self.ns_px_samples, self.ns_pmra_samples, self.ns_pmdec_samples = self.generate_multivariate_clones(self.target)

        if self.target['radial_velocity_error'] == 17.32 or self.target['radial_velocity_error'] == 0.0:
            self.ns_rv_samples = np.random.normal(0.0, 101.0, self.N)
        else:
            self.ns_rv_samples = np.random.normal(self.target['radial_velocity'], self.target['radial_velocity_error'], self.N)

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
        axes_c = np.sqrt(np.maximum(evals_c, 1e-100)) * self.n_sigma
        
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
        axes_t = np.sqrt(np.maximum(evals_t, 1e-100)) * self.n_sigma
        
        points_c = np.vstack((x_c - xt_mean, y_c - yt_mean, z_c - zt_mean))
        points_c_rot = np.dot(evecs_t.T, points_c)
        
        frac_cand = np.sum((points_c_rot[0]/axes_t[0])**2 + 
                           (points_c_rot[1]/axes_t[1])**2 + 
                           (points_c_rot[2]/axes_t[2])**2 < 1.0) / self.N
        
        final_prob = (frac_target * frac_cand)
        
        # Physical distance between the centers (to see how many parsecs apart they are)
        clone_distances_pc = np.sqrt((x_c - x_t)**2 + (y_c - y_t)**2 + (z_c - z_t)**2)
        
        min_dist_median = np.median(clone_distances_pc)
        min_dist_absolute = np.min(clone_distances_pc)
        
        return final_prob, min_dist_median, min_dist_absolute

    def run_global_scan(self, df):
        """Runs the global probability scan for all stars."""
        print(f"\nStarting global Monte Carlo scan (N={self.N}) for {len(df)} stars...")
        probs, dists_median, dists_absolute = [], [], []
        
        total_stars = len(df)
        start_time = time.time()

        for i, (index, row) in enumerate(df.iterrows()):
            np.random.seed(i)

            ra_rad_arr, dec_rad_arr, px_samples, pmra_samples, pmdec_samples = self.generate_multivariate_clones(row)
            tau_samples = np.random.normal(loc=self.tau_mean, scale=self.tau_error, size=self.N)
            cand_rv_samples = np.random.normal(row['radial_velocity'], row['radial_velocity_error'], self.N)

            ra_past_rad, dec_past_rad = self.ep.propagate_pos(
                ra_rad_arr, dec_rad_arr, px_samples, 
                pmra_samples, pmdec_samples, cand_rv_samples, 2016.0, 2016.0 - tau_samples
            )
            
            ns_ra_past_rad, ns_dec_past_rad = self.ep.propagate_pos(
                self.ns_ra_arr, self.ns_dec_arr, self.ns_px_samples, 
                self.ns_pmra_samples, self.ns_pmdec_samples, self.ns_rv_samples, 2016.0, 2016.0 - tau_samples
            )


            prob, dist_med, dist_abs = self.calculate_overlap_3d_ellipsoid(ra_past_rad, dec_past_rad, px_samples, ns_ra_past_rad, ns_dec_past_rad, self.ns_px_samples)
            probs.append(prob)
            dists_median.append(dist_med)
            dists_absolute.append(dist_abs)

            elapsed_time = time.time() - start_time
            print(f"\rSimulating: {i + 1}/{total_stars} stars | Time: {elapsed_time:.1f} s", end="", flush=True)
                
        print()
            
        df['crossing_probability'] = probs
        df['min_distance_median_pc'] = dists_median
        df['min_distance_absolute_pc'] = dists_absolute
        return df

    def run_time_series(self, candidates, eval_times, candidates_number):
        """Runs the exact time-step scanner for top candidates."""
        print(f"\nCalculating time series probability for {len(candidates)} top candidates...")
        
        total_stars = len(candidates)
        start_time = time.time()

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
            
            ra_rad_arr, dec_rad_arr, px_samples, pmra_samples, pmdec_samples = self.generate_multivariate_clones(row)

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

                prob, _, _ = self.calculate_overlap_3d_ellipsoid(ra_past_rad, dec_past_rad, px_samples, ns_ra_past_rad, ns_dec_past_rad, self.ns_px_samples)

                time_probs.append(prob)

                if prob > max_prob_for_star:
                    max_prob_for_star = prob
                    age_at_max = t
                    
            prob_over_time[row['source_id']] = time_probs
            peak_probabilities.append(max_prob_for_star)
            peak_ages.append(age_at_max)

            elapsed_time = time.time() - start_time
            print(f"\rProbability for: {idx + 1}/{total_stars} stars | Time: {elapsed_time:.1f} s", end="", flush=True)

        candidates['peak_probability'] = peak_probabilities
        candidates['kinematic_age'] = peak_ages
        
        candidates = candidates.sort_values(by=['peak_probability', 'min_distance_median_pc'], ascending=[False, True])
        return candidates.head(candidates_number), prob_over_time

class Visualizer:
    def __init__(self, target_id, top_candidates, simulator):
        """Initializes the visualizer and generates consistent colors for the top candidates."""
        self.simulator = simulator
        self.target_id = target_id
        self.top_candidates = top_candidates
        
        self.colors = cm.rainbow(np.linspace(0, 1, len(self.top_candidates)))
        np.random.seed(99) 
        np.random.shuffle(self.colors)

    def export_excel(self, prob_over_time, eval_times, all_candidates, filename, time_div=1000, time_unit="kyr"):
        """Exports the detailed results to an Excel file."""
        print("\nGenerating Excel report...")

        cols_info = [
            'source_id', 'peak_probability', 'kinematic_age', 'crossing_probability', 'min_distance_median_pc',
            'ra', 'dec', 'distance_pc', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'v_transverse', 'radial_velocity', 'radial_velocity_error',
            'abs_mag_g', 'bp_rp_color', 'A_V'
        ]
        df_info = all_candidates[cols_info].copy()
        df_info.sort_values(by='peak_probability', ascending=False, inplace=True)

        df_info.rename(columns={
            'source_id': 'Gaia DR3 Source ID',
            'peak_probability': 'Peak Prob.',
            'kinematic_age': 'Kinematic age',
            'min_distance_median_pc': 'Min Dist Median (pc)',
            'crossing_probability': 'Global Hit Prob.',
            'ra': 'RA (deg)',
            'dec': 'DEC (deg)',
            'distance_pc': 'Dist (pc)',
            'pmra': 'PMra',
            'pmra_error': 'PMra_err', 
            'pmdec': 'PMdec', 
            'pmdec_error': 'PMdec_err',
            'v_transverse': 'V_T (km/s)',
            'radial_velocity': 'RV (km/s)',
            'radial_velocity_error': 'RV_err (km/s)',
            'abs_mag_g': 'Abs Mag (Mg)',
            'bp_rp_color': 'Color (BP-RP)0',
            'A_V': 'Extinction (Av)'
        }, inplace=True)

        prob_data = {'Gaia DR3 Source ID': self.top_candidates['source_id'].values}

        for t_idx, t_val in enumerate(eval_times):
            col_name = f"{t_val/time_div:.1f} {time_unit}"
            prob_data[col_name] = [prob_over_time[sid][t_idx] for sid in self.top_candidates['source_id']]

        df_probs = pd.DataFrame(prob_data)

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_info.to_excel(writer, sheet_name='Astrophysical_Details', index=False)
            df_probs.to_excel(writer, sheet_name='Time_Series_Probs', index=False)

        print(f"Excel file created: {filename}")

    def plot_time_series(self, prob_over_time, eval_times, tau_mean, tau_error, time_div, time_unit):
        """Plots the kinematic intersection probability over time."""
        plt.figure(figsize=(10, 6))

        for idx, (index, row) in enumerate(self.top_candidates.iterrows()):
            source = row['source_id']
            probs = prob_over_time[source]
            color = self.colors[idx] 
            
            plt.plot(eval_times / time_div, probs, linewidth=2, marker='.', color=color, label=f"{idx + 1}: {source}")

        plt.axvline(x=tau_mean/time_div, color='black', linestyle='--', linewidth=2, label='Estimated Age')

        #plt.title('Kinematic Intersection Probability over Time', fontsize=14, fontweight='bold')
        plt.xlabel(f'Time into the past ({time_unit})', fontsize=12)
        plt.ylabel('Overlap Probability (Hits / N)', fontsize=12)

        plt.yscale('log')

        plt.xlim((tau_mean-tau_error)/time_div, (tau_mean+tau_error)/time_div)
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
        #plt.scatter(t_col, t_mg, color='yellow', edgecolor='black', s=500, marker='.', zorder=10, label=f'Pulsar: {self.target_id}')
        
        plt.gca().invert_yaxis()
        #plt.title('HR Diagram', fontsize=14, fontweight='bold')
        plt.xlabel('Intrinsic Color $(BP - RP)_0$', fontsize=12)
        plt.ylabel('Absolute Magnitude $M_G$', fontsize=12)
        plt.xlim(-1, 6)
        plt.ylim(18, -8)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9, framealpha=0.9, edgecolor='black')

        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
    
    def plot_fits_map(self, fits_file, ep, target_star, tau_mean, tau_error, N, number_sigma, center_ra, center_dec, window_deg=None):
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

        lon = ax1.coords[0]
        lat = ax1.coords[1]
        lon.set_major_formatter('d.dd')
        lat.set_major_formatter('d.dd')
        ax1.set_xlabel('Right Ascension (deg)')
        ax1.set_ylabel('Declination (deg)')

        vmin_auto = np.nanpercentile(data, 5)
        vmax_auto = np.nanpercentile(data, 97)
        ax1.imshow(data, vmin=vmin_auto, vmax=vmax_auto, cmap='Greys', origin='lower')
        #Gray -> black

        # Pulsar Data
        ns_ra, ns_dec = target_star['ra'], target_star['dec']
        ns_parallax, ns_rv = target_star['parallax'], target_star['radial_velocity']
        ns_pmra, ns_pmdec = target_star['pmra'], target_star['pmdec']
        ns_pmra_err, ns_pmdec_err = target_star['pmra_error'], target_star['pmdec_error']

        # Pulsar arrays Monte Carlo
        np.random.seed(888)

        draw_ra_rad_arr = self.simulator.ns_ra_arr
        draw_dec_rad_arr = self.simulator.ns_dec_arr
        draw_px_arr = self.simulator.ns_px_samples
        draw_pmra_samples = self.simulator.ns_pmra_samples
        draw_pmdec_samples = self.simulator.ns_pmdec_samples

        if target_star['radial_velocity_error'] == 17.32 or target_star['radial_velocity_error'] == 0.0:
            draw_rv_arr = np.random.normal(0.0, 101.0, N)
        else:
            draw_rv_arr = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)

        p_ra_s, p_dec_s = ep.propagate_pos(draw_ra_rad_arr, draw_dec_rad_arr, draw_px_arr, draw_pmra_samples, draw_pmdec_samples, draw_rv_arr, 2016.0, 2016 - tau_mean - tau_error)
        p_ra_e, p_dec_e = ep.propagate_pos(draw_ra_rad_arr, draw_dec_rad_arr, draw_px_arr, draw_pmra_samples, draw_pmdec_samples, draw_rv_arr, 2016.0, 2016 - tau_mean + tau_error)
        p_ra_c, p_dec_c = ep.propagate_pos(draw_ra_rad_arr, draw_dec_rad_arr, draw_px_arr, draw_pmra_samples, draw_pmdec_samples, draw_rv_arr, 2016.0, 2016 - tau_mean)

        p_px_x_cloud, p_px_y_cloud = [], []
        
        for i in range(N):
            px_s_x, px_s_y = get_pixel(np.degrees(p_ra_s[i]), np.degrees(p_dec_s[i]))
            px_e_x, px_e_y = get_pixel(np.degrees(p_ra_e[i]), np.degrees(p_dec_e[i]))
            px_c_x, px_c_y = get_pixel(np.degrees(p_ra_c[i]), np.degrees(p_dec_c[i]))
            
            # Monte Carlo line cut
            ax1.plot([float(px_s_x), float(px_e_x)], [float(px_s_y), float(px_e_y)], '-', color='yellow', alpha=0.03, zorder=2)
            
            p_px_x_cloud.append(float(px_c_x))
            p_px_y_cloud.append(float(px_c_y))

        # Cloud of the Pulsar
        ax1.plot(p_px_x_cloud, p_px_y_cloud, '.', color='yellow', markersize=2, alpha=0.5, zorder=8)

        ns_ra_mean, ns_dec_mean = ep.propagate_pos(np.radians(ns_ra), np.radians(ns_dec), ns_parallax, ns_pmra, ns_pmdec, ns_rv, 2016.0, 2016.0 - tau_mean)
        p_mean_x, p_mean_y = get_pixel(np.degrees(ns_ra_mean), np.degrees(ns_dec_mean))
        ax1.plot(p_mean_x, p_mean_y, '.', color='yellow', markersize=12, markeredgecolor='black', label=f'Pulsar (Est. Age: {tau_mean/1000:.0f} kyr)', zorder=10)

        target_curr_x, target_curr_y = get_pixel(ns_ra, ns_dec)
        target_curr_x, target_curr_y = float(target_curr_x), float(target_curr_y)
        
        ax1.plot([p_mean_x, target_curr_x], [p_mean_y, target_curr_y], '-', color='yellow', linewidth=2, zorder=5)
        
        ax1.plot(target_curr_x, target_curr_y, '.', color='white', markersize=14, markeredgecolor='black', label='Pulsar (Today)', zorder=10)
        
        for i in range(N):
             ax1.plot([target_curr_x, p_px_x_cloud[i]], [target_curr_y, p_px_y_cloud[i]], '-', color='yellow', alpha=0.02, linewidth=1, zorder=2)

        
        for idx, (index, row) in enumerate(self.top_candidates.iterrows()):
            color = self.colors[idx]
            
            ra, dec = row['ra'], row['dec']
            pmra, e_pmra = row['pmra'], row['pmra_error']
            pmdec, e_pmdec = row['pmdec'], row['pmdec_error']

            np.random.seed(index)

            ra_rad_arr, dec_rad_arr, px_arr, pmra_samples, pmdec_samples = self.simulator.generate_multivariate_clones(row)
            
            rv_samples = np.random.normal(loc=row['radial_velocity'], scale=row['radial_velocity_error'], size=N)
            
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

            ax1.plot([x_central, X_float], [y_central, Y_float], '-', color=color, linewidth=2, zorder=5)
            ax1.plot(x_central, y_central, 'X', color=color, markersize=8, markeredgecolor='black', zorder=6)
            
            label_text = f'{idx+1}: {row["source_id"]}'
            if row["source_id"] == self.target_id:
                label_text += ' *'
                
            ax1.plot(X_float, Y_float, 'o', color=color, markersize=8, markeredgecolor='white', label=label_text, zorder=7)
            ax1.text(X_float + 0.5, Y_float, str(idx + 1), color='black', fontsize=8, fontweight='bold', zorder=7)

            # ELLIPSOIDS AT THE EXACT MOMENT OF CROSSING ---
            crossing_age = row['kinematic_age']
            
            if crossing_age > 0:
                
                # Propagate the candidate in the exact year of crossbreeding
                ra_cruce_rad, dec_cruce_rad = ep.propagate_pos(
                    ra_rad_arr, dec_rad_arr, px_arr, 
                    pmra_samples, pmdec_samples, rv_samples, 2016.0, 2016.0 - crossing_age
                )
                c_px_x, c_px_y = get_pixel(np.degrees(ra_cruce_rad), np.degrees(dec_cruce_rad))

                # Candidate cloud
                ax1.plot(c_px_x, c_px_y, '.', color=color, markersize=1, alpha=0.5, zorder=7)
                
                # Pulsar cloud
                #ax1.plot(p_px_x, p_px_y, '.', color='yellow', markersize=1, alpha=0.5, zorder=7)
            
        center_x, center_y = get_pixel(center_ra, center_dec)
        center_x, center_y = float(center_x), float(center_y)

        #ax1.set_xlim([0, data.shape[1]])
        #ax1.set_ylim([0, data.shape[0]])

        if window_deg is not None:
            # proj_plane_pixel_scales saca cuántos grados mide un píxel en este FITS
            pixel_scale = proj_plane_pixel_scales(w)[0] 
            window_pixels = (window_deg / pixel_scale) / 2.0
            
            ax1.set_xlim([center_x - window_pixels, center_x + window_pixels])
            ax1.set_ylim([center_y - window_pixels, center_y + window_pixels])


        #plt.title('Kinematic Intersection (Moving Target)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=8)
        plt.show(block=False)

    def create_gif(self, df, ep, target_star, eval_times, N, number_sigma, output_folder, gif_filename, center_ra_past, center_dec_past):
        """Creates the animated GIF of the trajectories over time."""

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        time_centers = eval_times
        time_window = abs(eval_times[1] - eval_times[0]) * 2 if len(eval_times) > 1 else 1000

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
        t_min = np.min(time_centers)

        ns_ra_max_rad, ns_dec_max_rad = ep.propagate_pos(
            np.radians(ns_ra), np.radians(ns_dec), ns_parallax, 
            ns_pmra, ns_pmdec, ns_rv, 2016.0, 2016.0 - t_max
        )
        ns_ra_max = np.degrees(ns_ra_max_rad)
        ns_dec_max = np.degrees(ns_dec_max_rad)

        ns_ra_min_rad, ns_dec_min_rad = ep.propagate_pos(
            np.radians(ns_ra), np.radians(ns_dec), ns_parallax, 
            ns_pmra, ns_pmdec, ns_rv, 2016.0, 2016.0 - t_min
        )
        ns_ra_min = np.degrees(ns_ra_min_rad)
        ns_dec_min = np.degrees(ns_dec_min_rad)

        ns_ra_cloud_max_rad, ns_dec_cloud_max_rad = ep.propagate_pos(
            ns_ra_rad_arr, ns_dec_rad_arr, ns_px_samples, 
            ns_pmra_samples, ns_pmdec_samples, ns_rv_arr, 
            2016.0, 2016.0 - t_max
        )
        ns_ra_cloud_max = np.degrees(ns_ra_cloud_max_rad)
        ns_dec_cloud_max = np.degrees(ns_dec_cloud_max_rad)

        global_ra_min = min(ns_ra_min, ns_ra_max, np.min(ns_ra_cloud_max))
        global_ra_max = max(ns_ra_min, ns_ra_max, np.max(ns_ra_cloud_max))
        global_dec_min = min(ns_dec_min, ns_dec_max, np.min(ns_dec_cloud_max))
        global_dec_max = max(ns_dec_min, ns_dec_max, np.max(ns_dec_cloud_max))

        center_ra_global = (global_ra_max + global_ra_min) / 2.0
        center_dec_global = (global_dec_max + global_dec_min) / 2.0
        
        ra_span_global = global_ra_max - global_ra_min
        dec_span_global = global_dec_max - global_dec_min
        zoom_window_global = max(ra_span_global, dec_span_global) / 2.0 * 1.2

        for frame_idx, t_mid in enumerate(time_centers):
            t_kyr = t_mid / 1000.0

            if t_kyr % 20 == 0:
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

            #ax1.plot([ns_ra, ns_ra_max], [ns_dec, ns_dec_max], '--', color='yellow', alpha=0.4, linewidth=1)
            ax1.plot([ns_ra_min, ns_ra_max], [ns_dec_min, ns_dec_max], '--', color='yellow', alpha=0.4, linewidth=1)
            
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

            ns_ra_min_frame = np.min(ns_ra_cloud)
            ns_ra_max_frame = np.max(ns_ra_cloud)
            ns_dec_min_frame = np.min(ns_dec_cloud)
            ns_dec_max_frame = np.max(ns_dec_cloud)
            
            ra_center = (ns_ra_max_frame + ns_ra_min_frame) / 2.0
            dec_center = (ns_dec_max_frame + ns_dec_min_frame) / 2.0
            
            ra_span_frame = ns_ra_max_frame - ns_ra_min_frame
            dec_span_frame = ns_dec_max_frame - ns_dec_min_frame
            
            zoom_window_dynamic = max(ra_span_frame, dec_span_frame) / 2.0 * 1.2
            
            if zoom_window_dynamic < 0.005:
                zoom_window_dynamic = 0.005

            ax1.set_xlim(center_ra_global + zoom_window_global, center_ra_global - zoom_window_global) 
            ax1.set_ylim(center_dec_global - zoom_window_global, center_dec_global + zoom_window_global)


            ax1.set_xlabel('Right Ascension (deg)')
            ax1.set_ylabel('Declination (deg)')
            ax2.set_xlabel('Right Ascension (deg)')
            
            ax1.grid(True, linestyle='--', alpha=0.4)
            ax2.grid(True, linestyle='--', alpha=0.4)
            
            ax1.legend(loc='lower left', fontsize=8, framealpha=0.8, edgecolor='black')
            
            plt.tight_layout()
            filename = f'frame_{frame_idx:04d}.png'
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
                duration=40,  
                loop=0         
            )
            print(f"GIF created: {gif_filename}")
    
    def plot_3d_projections_2d(self, ep, target_star, candidate_row, N=5000):
        """Plots 2D orthogonal projections (X-Y, X-Z, Y-Z) of the 3D encounter."""
        print(f"\nGenerating 2D Spatial Projections for {candidate_row['source_id']}...")
        
        t_enc = candidate_row['kinematic_age']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Pulsar vs 1st Candidate ({t_enc/1000:.0f} kyr ago)', fontsize=16, fontweight='bold')
        
        def get_past_distance(px_mas, rv_kms, t_years):
            d_pres_pc = 1000.0 / px_mas
            return d_pres_pc - (rv_kms * t_years * 1.02271e-6)
            
        def to_xyz(ra_rad, dec_rad, d_pc):
            x = d_pc * np.cos(dec_rad) * np.cos(ra_rad)
            y = d_pc * np.cos(dec_rad) * np.sin(ra_rad)
            z = d_pc * np.sin(dec_rad)
            return x, y, z

        np.random.seed(888)

        p_ra_arr, p_dec_arr, p_px_arr, p_pmra_arr, p_pmdec_arr = self.simulator.generate_multivariate_clones(target_star, custom_N=N)

        if target_star['radial_velocity_error'] == 17.32 or target_star['radial_velocity_error'] == 0.0:
            p_rv_arr = np.random.normal(0.0, 101.0, N)
        else:
            p_rv_arr = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)
            
        p_ra_past, p_dec_past = ep.propagate_pos(p_ra_arr, p_dec_arr, p_px_arr, p_pmra_arr, p_pmdec_arr, p_rv_arr, 2016.0, 2016.0 - t_enc)
        p_d_past = get_past_distance(p_px_arr, p_rv_arr, t_enc)
        p_x, p_y, p_z = to_xyz(p_ra_past, p_dec_past, p_d_past)
        
        np.random.seed(42)

        c_ra_arr, c_dec_arr, c_px_arr, c_pmra_arr, c_pmdec_arr = self.simulator.generate_multivariate_clones(candidate_row, custom_N=N)

        c_rv_arr = np.random.normal(candidate_row['radial_velocity'], candidate_row['radial_velocity_error'], N)
        
        c_ra_past, c_dec_past = ep.propagate_pos(c_ra_arr, c_dec_arr, c_px_arr, c_pmra_arr, c_pmdec_arr, c_rv_arr, 2016.0, 2016.0 - t_enc)
        c_d_past = get_past_distance(c_px_arr, c_rv_arr, t_enc)
        c_x, c_y, c_z = to_xyz(c_ra_past, c_dec_past, c_d_past)
        
        labels = [('X (pc)', 'Y (pc)'), ('X (pc)', 'Z (pc)'), ('Y (pc)', 'Z (pc)')]
        data_p = [(p_x, p_y), (p_x, p_z), (p_y, p_z)]
        data_c = [(c_x, c_y), (c_x, c_z), (c_y, c_z)]
        
        for i, ax in enumerate(axes):
            # Nubes de puntos
            ax.scatter(data_p[i][0], data_p[i][1], color='orange', alpha=0.03, s=2, zorder=1)
            ax.scatter(data_c[i][0], data_c[i][1], color='blue', alpha=0.03, s=2, zorder=2)
            
            # Centroides (posición media exacta)
            ax.plot(np.mean(data_p[i][0]), np.mean(data_p[i][1]), '.', color="#e0c758", markersize=14, markeredgecolor='black', zorder=4)
            ax.plot(np.mean(data_c[i][0]), np.mean(data_c[i][1]), '.', color="#86d3ff", markersize=10, markeredgecolor='black', zorder=3)
            
            ax.set_xlabel(labels[i][0], fontsize=12)
            ax.set_ylabel(labels[i][1], fontsize=12)
            
            ax.set_aspect('equal', adjustable='datalim') 
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if i == 0:
                import matplotlib.lines as mlines
                puls_line = mlines.Line2D([], [], color="#e0c758", marker='.', linestyle='None', markersize=10, markeredgecolor='black', label='Pulsar Cloud')
                cand_line = mlines.Line2D([], [], color="#86d3ff", marker='.', linestyle='None', markersize=8, markeredgecolor='black', label='Candidate Cloud')
                ax.legend(handles=[puls_line, cand_line], loc='best')
                
        min_x = min(np.percentile(p_x, 1), np.percentile(c_x, 1))
        max_x = max(np.percentile(p_x, 99), np.percentile(c_x, 99))
        
        min_y = min(np.percentile(p_y, 1), np.percentile(c_y, 1))
        max_y = max(np.percentile(p_y, 99), np.percentile(c_y, 99))
        
        min_z = min(np.percentile(p_z, 1), np.percentile(c_z, 1))
        max_z = max(np.percentile(p_z, 99), np.percentile(c_z, 99))

        margin = 15
        
        axes[0].set_xlim([min_x - margin, max_x + margin])
        axes[0].set_ylim([min_y - margin, max_y + margin])
        
        axes[1].set_xlim([min_x - margin, max_x + margin])
        axes[1].set_ylim([min_z - margin, max_z + margin])
        
        axes[2].set_xlim([min_y - margin, max_y + margin])
        axes[2].set_ylim([min_z - margin, max_z + margin])

        plt.tight_layout()
        plt.show(block=False)
    
    def save_meta_parameters(self, pulsar_id, best_cand, output_file="pulsar_population.csv"):
        """Saves or updates the meta-parameters of the best candidate in a master CSV."""
        import os
        import pandas as pd
        
        system_type_val = 'Check SIMBAD'
        
        if os.path.isfile(output_file):
            df_existing = pd.read_csv(output_file)
            df_existing['Pulsar_ID'] = df_existing['Pulsar_ID'].astype(str)
            
            if str(pulsar_id) in df_existing['Pulsar_ID'].values:
                old_val = df_existing.loc[df_existing['Pulsar_ID'] == str(pulsar_id), 'System_Type'].values[0]
                if pd.notna(old_val):
                    system_type_val = old_val

        meta_data = {
            'Pulsar_ID': str(pulsar_id),
            'Best_Candidate_Gaia_ID': best_cand['source_id'],
            'Peak_Probability': best_cand['peak_probability'],
            'Kinematic_Age_kyr': best_cand['kinematic_age'] / 1000.0,
            'Min_Dist_median_pc': best_cand['min_distance_median_pc'],
            'Abs_Mag_Mg': best_cand.get('Abs Mag (Mg)', best_cand.get('phot_g_mean_mag', 'N/A')),
            'Color_BP_RP': best_cand.get('Color (BP-RP)', best_cand.get('bp_rp', 'N/A')),
            'V_Transverse_kms': best_cand.get('V_T (km/s)', best_cand.get('v_transverse', 'N/A')),
            'Radial_Vel_kms': best_cand.get('radial_velocity', 'N/A'),
            'RV_Error': best_cand.get('radial_velocity_error', 'N/A'),
            'System_Type': system_type_val 
        }

        df_new = pd.DataFrame([meta_data])

        if not os.path.isfile(output_file):
            df_new.to_csv(output_file, index=False)
        else:
            df_existing = df_existing[df_existing['Pulsar_ID'] != str(pulsar_id)]
                
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(output_file, index=False)
            
    
    def plot_3d_interactive_plotly(self, ep, target_star, candidate_row, N=3000):
        """Generates an interactive 3D HTML plot using Plotly to explore the encounter."""
        print(f"\nGenerating Interactive 3D Plot for {candidate_row['source_id']}...")
        
        t_enc = candidate_row['kinematic_age']
        
        def get_past_distance(px_mas, rv_kms, t_years):
            d_pres_pc = 1000.0 / px_mas
            return d_pres_pc - (rv_kms * t_years * 1.02271e-6)
            
        def to_xyz(ra_rad, dec_rad, d_pc):
            x = d_pc * np.cos(dec_rad) * np.cos(ra_rad)
            y = d_pc * np.cos(dec_rad) * np.sin(ra_rad)
            z = d_pc * np.sin(dec_rad)
            return x, y, z

        np.random.seed(888)
        p_ra_arr = np.full(N, np.radians(target_star['ra']))
        p_dec_arr = np.full(N, np.radians(target_star['dec']))
        p_px_arr = np.random.normal(target_star['parallax'], target_star['parallax_error'], N)
        p_pmra_arr = np.random.normal(target_star['pmra'], target_star['pmra_error'], N)
        p_pmdec_arr = np.random.normal(target_star['pmdec'], target_star['pmdec_error'], N)
        
        if target_star['radial_velocity_error'] == 17.32 or target_star['radial_velocity_error'] == 0.0:
            p_rv_arr = np.random.normal(0.0, 101.0, N)
        else:
            p_rv_arr = np.random.normal(target_star['radial_velocity'], target_star['radial_velocity_error'], N)
            
        p_ra_past, p_dec_past = ep.propagate_pos(p_ra_arr, p_dec_arr, p_px_arr, p_pmra_arr, p_pmdec_arr, p_rv_arr, 2016.0, 2016.0 - t_enc)
        p_d_past = get_past_distance(p_px_arr, p_rv_arr, t_enc)
        p_x, p_y, p_z = to_xyz(p_ra_past, p_dec_past, p_d_past)
        
        np.random.seed(42)
        c_ra_arr = np.full(N, np.radians(candidate_row['ra']))
        c_dec_arr = np.full(N, np.radians(candidate_row['dec']))
        c_px_arr = np.random.normal(candidate_row['parallax'], candidate_row['parallax_error'], N)
        c_pmra_arr = np.random.normal(candidate_row['pmra'], candidate_row['pmra_error'], N)
        c_pmdec_arr = np.random.normal(candidate_row['pmdec'], candidate_row['pmdec_error'], N)
        c_rv_arr = np.random.normal(candidate_row['radial_velocity'], candidate_row['radial_velocity_error'], N)
        
        c_ra_past, c_dec_past = ep.propagate_pos(c_ra_arr, c_dec_arr, c_px_arr, c_pmra_arr, c_pmdec_arr, c_rv_arr, 2016.0, 2016.0 - t_enc)
        c_d_past = get_past_distance(c_px_arr, c_rv_arr, t_enc)
        c_x, c_y, c_z = to_xyz(c_ra_past, c_dec_past, c_d_past)
        
        c_x_mean, c_y_mean, c_z_mean = np.mean(c_x), np.mean(c_y), np.mean(c_z)

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=p_x, y=p_y, z=p_z,
            mode='markers',
            marker=dict(size=2, color='orange', opacity=0.15),
            name='Pulsar Cloud'
        ))

        fig.add_trace(go.Scatter3d(
            x=c_x, y=c_y, z=c_z,
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.15),
            name='Candidate Cloud'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers+text',
            marker=dict(size=5, color='green', symbol='cross'),
            text=["Earth"],
            textposition="bottom center",
            name='Earth (Solar System)'
        ))

        fig.add_trace(go.Scatter3d(
            x=[0, c_x_mean], y=[0, c_y_mean], z=[0, c_z_mean],
            mode='lines',
            line=dict(color='white', width=2, dash='dash'),
            name='Vision Line'
        ))

        fig.update_layout(
            title=f"{t_enc/1000:.0f} kyr ago",
            scene=dict(
                xaxis_title='Heliocentric X (pc)',
                yaxis_title='Heliocentric Y (pc)',
                zaxis_title='Heliocentric Z (pc)',
                aspectmode='data',
                xaxis=dict(showbackground=True, backgroundcolor="rgb(20, 20, 20)", gridcolor="white"),
                yaxis=dict(showbackground=True, backgroundcolor="rgb(20, 20, 20)", gridcolor="white"),
                zaxis=dict(showbackground=True, backgroundcolor="rgb(20, 20, 20)", gridcolor="white")
            ),
            paper_bgcolor="black",
            font=dict(color="white"),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.show()

def load_pulsar(csv_path, pulsar_id):
    
    try:
        df_targets = pd.read_csv(csv_path)
        pulsar_row = df_targets[df_targets['Pulsar_ID'] == pulsar_id].iloc[0]
        
        target_star = pulsar_row.to_dict()
        
        pulsar_age = pulsar_row['age_kyr'] * 1000
        pulsar_age_error = pulsar_row['age_error_kyr'] * 1000

        print(f"Data loaded for {pulsar_id}. Estimated age: {pulsar_age} years.")
        return target_star, pulsar_age, pulsar_age_error
        
    except IndexError:
        print(f"Pulsar '{pulsar_id}' not found in {csv_path}.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"File {csv_path} does not exist.")
        sys.exit(1)


if __name__ == "__main__":
    np.random.seed(42)

    PULSAR_NAME = 'B0833-45'
    CANDIDATES_NUMBER = 10
    FILTERS = 0 # 0: Without filter - 1: With filter

    BASE_DIR = Path(__file__).resolve().parent
    PULSAR_DATA_DIR = BASE_DIR / "pulsars" / PULSAR_NAME

    FITS_FILE = str(PULSAR_DATA_DIR / "fits" / f"{PULSAR_NAME}.fits")
    CSV_FILE  = str(PULSAR_DATA_DIR / "starsDB" / f"{PULSAR_NAME}.csv")
    AV_FILE  = str(PULSAR_DATA_DIR / f"Av_profile_{PULSAR_NAME}.txt")
    TARGETS_CSV = str(BASE_DIR / "pulsar_targets.csv")

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"Gaia catalogue is missing: {CSV_FILE}")
    if not os.path.exists(FITS_FILE):
        raise FileNotFoundError(f"FITS file is missing: {FITS_FILE}")
    if not os.path.exists(AV_FILE):
        raise FileNotFoundError(f"Av profile file is missing: {AV_FILE}")
    
    target_star, mean_age, age_error = load_pulsar(TARGETS_CSV, PULSAR_NAME)
    
    TAU_PARAMS = {'mean': mean_age, 'error': age_error, 'min': mean_age - age_error, 'max': mean_age + age_error}
    SIM_PARAMS = {'N': 1000, 'sigma': 3}

    TIME_STEP_YRS = target_star['time_steps']  
    TIME_DIVISOR = target_star['time_steps']       
    TIME_LABEL = target_star['time_label'] 

    # CALCULATE PAST CENTER FOR THE FITS
    pulsar_to_past = EpochPropagation().propagate_pos(
        np.radians(target_star['ra']), np.radians(target_star['dec']), target_star['parallax'],
        target_star['pmra'], target_star['pmdec'], target_star['radial_velocity'], 2016.0, 2016.0 - TAU_PARAMS['mean']
    )
    center_ra_past, center_dec_past = pulsar_to_past[0], pulsar_to_past[1]
    print(f"Coords: {np.degrees(center_ra_past):.8f} {np.degrees(center_dec_past):.8f} deg")
    #sys.exit()
    
    catalog = StarCatalog(CSV_FILE, AV_FILE)
    filtered_stars = catalog.apply_filters(filter=FILTERS)

    simulator = KinematicSimulator(target_star, TAU_PARAMS, SIM_PARAMS)
    
    analyzed_stars = simulator.run_global_scan(filtered_stars)

    candidates = analyzed_stars[analyzed_stars['crossing_probability'] > 0].copy()

    if len(candidates) == 0:
        candidates = analyzed_stars[analyzed_stars['min_distance_median_pc'] < 100.0].copy()

    target_star['abs_mag_g'] = 0.0 
    target_star['bp_rp_color'] = 0.0
    
    if not candidates.empty:
        eval_times = np.arange(TAU_PARAMS['min'], TAU_PARAMS['max'], TIME_STEP_YRS)
        top_candidates, prob_over_time = simulator.run_time_series(candidates, eval_times, CANDIDATES_NUMBER)
        
        print("\nTOP CANDIDATES FOUND:")
        print(top_candidates[['source_id', 'peak_probability', 'kinematic_age', 'min_distance_median_pc']])
        
        viz = Visualizer(PULSAR_NAME, top_candidates, simulator)

        #best_cand = top_candidates.iloc[0]
        #viz.plot_3d_encounter(target_star, best_cand, best_cand['kinematic_age'])

        best_cand = top_candidates.iloc[0]
        best_age = best_cand['kinematic_age']
        
        #viz.plot_3d_ellipsoids_snapshot(target_star, top_candidates, best_age, n_sigma=2.0)

        path_excel = str(PULSAR_DATA_DIR / "Top_Candidates.xlsx")
        path_gif_folder = str(PULSAR_DATA_DIR / "runaway_frames")
        path_gif_file = str(PULSAR_DATA_DIR / "runaway_evolution.gif")
        
        viz.export_excel(prob_over_time, eval_times, candidates, filename=path_excel, time_div=TIME_DIVISOR, time_unit=TIME_LABEL)
        viz.plot_time_series(prob_over_time, eval_times, TAU_PARAMS['mean'], TAU_PARAMS['error'], TIME_DIVISOR, TIME_LABEL)
        viz.plot_hr_diagram(filtered_stars, target_star)
        viz.plot_fits_map(FITS_FILE, simulator.ep, target_star, TAU_PARAMS['mean'], TAU_PARAMS['error'], SIM_PARAMS['N'], SIM_PARAMS['sigma'], center_ra_past, center_dec_past)
        best_cand = top_candidates.iloc[0]
        viz.plot_3d_projections_2d(simulator.ep, target_star, best_cand, N=10000)
        #viz.plot_3d_interactive_plotly(simulator.ep, target_star, best_cand, N=3000)
        #viz.save_meta_parameters(PULSAR_NAME, best_cand)

        plt.show(block=False)
        plt.pause(0.1)

        viz.create_gif(filtered_stars, simulator.ep, target_star, eval_times, SIM_PARAMS['N'], SIM_PARAMS['sigma'], output_folder=path_gif_folder, gif_filename=path_gif_file, center_ra_past=center_ra_past, center_dec_past=center_dec_past)
        
        plt.show()
    else:
        print("\n No candidates found in this run.")