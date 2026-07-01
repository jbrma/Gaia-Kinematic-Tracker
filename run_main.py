import pandas as pd
import numpy as np
import os

# Import your classes directly from main_oop.py
from main_oop import StarCatalog, KinematicSimulator, load_pulsar, Visualizer

def run_batch():
    # Load the target list
    df_targets = pd.read_csv("pulsar_targets.csv")
    master_population = []
    
    # Global execution parameters
    SIM_PARAMS = {'N': 1000, 'sigma': 3}
    CANDIDATES_NUMBER = 10  # Adjust how many top candidates per pulsar to save

    # Loop through each pulsar
    for index, row in df_targets.iterrows():
        pulsar_name = row['Pulsar_ID']
        print(f"\n{'='*50}\nPROCESSING PULSAR: {pulsar_name}\n{'='*50}")

        csv_file = f"pulsars/{pulsar_name}/starsDB/{pulsar_name}.csv"
        av_file = f"pulsars/{pulsar_name}/Av_profile_{pulsar_name}.txt"

        # Skip if local Gaia CSV catalog is missing
        if not os.path.exists(csv_file):
            print(f"Skipping {pulsar_name} - Local CSV not found: {csv_file}")
            continue

        # Load intrinsic parameters
        target_star, mean_age, age_error = load_pulsar("pulsar_targets.csv", pulsar_name)
        tau_params = {'mean': mean_age, 'error': age_error, 'min': mean_age - age_error, 'max': mean_age + age_error}
        time_step_yrs = target_star['time_steps']

        # Dual execution: Without Filter (0) and With Filter (1)
        for apply_filter in [0, 1]:
            exec_type = "With H-R Filter" if apply_filter == 1 else "Without Filters"
            print(f"\n--- Execution: {exec_type} ---")
            
            try:
                # Instantiate catalog and apply chosen filter
                catalog = StarCatalog(csv_file, av_file)
                filtered_stars = catalog.apply_filters(filter=apply_filter)
                
                if len(filtered_stars) == 0:
                    print("No stars passed the filter. Skipping to next run.")
                    continue
                    
                # Run Monte Carlo global scan
                simulator = KinematicSimulator(target_star, tau_params, SIM_PARAMS)
                analyzed_stars = simulator.run_global_scan(filtered_stars)
                
                # Select viable survivors
                candidates = analyzed_stars[analyzed_stars['crossing_probability'] > 0].copy()
                if len(candidates) == 0:
                    candidates = analyzed_stars[analyzed_stars['min_distance_median_pc'] < 100.0].copy()
                    
                if not candidates.empty:
                    eval_times = np.arange(tau_params['min'], tau_params['max'], time_step_yrs)
                    # Extract the top candidates via time-series analysis
                    top_candidates, _ = simulator.run_time_series(candidates, eval_times, CANDIDATES_NUMBER)
                    
                    # Tag context data directly at the front of the DataFrame
                    top_candidates.insert(0, 'Pulsar_ID', pulsar_name)
                    top_candidates.insert(1, 'Execution_Type', exec_type)
                    
                    # Append full DataFrame (with all Gaia & custom parameters) to the master list
                    master_population.append(top_candidates)

                    print("\nStars analyzed and top candidates identified. Generating GIF visualization...")
                    if apply_filter == 1:
                        gif_folder = f"pulsars/{pulsar_name}/runaway_frames_filtered"
                        gif_file = f"pulsars/{pulsar_name}/runaway_evolution_filtered.gif"
                        os.makedirs(gif_folder, exist_ok=True)
                    else:
                        gif_folder = f"pulsars/{pulsar_name}/runaway_frames_unfiltered"
                        gif_file = f"pulsars/{pulsar_name}/runaway_evolution_unfiltered.gif"
                        os.makedirs(gif_folder, exist_ok=True)
                        
                    pulsar_to_past = simulator.ep.propagate_pos(
                        np.radians(target_star['ra']), np.radians(target_star['dec']), target_star['parallax'],
                        target_star['pmra'], target_star['pmdec'], target_star['radial_velocity'], 2016.0, 2016.0 - tau_params['mean']
                    )
                    center_ra_past, center_dec_past = pulsar_to_past[0], pulsar_to_past[1]

                    viz = Visualizer(pulsar_name, top_candidates, simulator)
                    viz.create_gif(
                        df=filtered_stars, 
                        ep=simulator.ep, 
                        target_star=target_star, 
                        eval_times=eval_times, 
                        N=SIM_PARAMS['N'], 
                        number_sigma=SIM_PARAMS['sigma'], 
                        output_folder=gif_folder, 
                        gif_filename=gif_file, 
                        center_ra_past=center_ra_past, 
                        center_dec_past=center_dec_past
                    )
                    print(f"  -> GIF saved in {gif_file}")

                else:
                    print("No viable candidates found.")
                    
            except Exception as e:
                print(f"Error processing {pulsar_name} ({exec_type}): {e}")

    if master_population:
        print("\nGenerating Master Excel...")
        df_master = pd.concat(master_population, ignore_index=True)
        
        # Rearrange essential columns to the front, keeping ALL others automatically
        key_cols = [
            'Pulsar_ID', 'Execution_Type', 'source_id', 'peak_probability', 
            'kinematic_age', 'min_distance_median_pc', 'min_distance_absolute_pc', 
            'crossing_probability'
        ]
        
        # Ensure we only sort columns that exist to prevent KeyErrors
        existing_key_cols = [c for c in key_cols if c in df_master.columns]
        other_cols = [c for c in df_master.columns if c not in existing_key_cols]
        
        df_master = df_master[existing_key_cols + other_cols]
        
        # Split data by execution type
        df_unfiltered = df_master[df_master['Execution_Type'] == "Without Filters"]
        df_filtered = df_master[df_master['Execution_Type'] == "With H-R Filter"]
        
        # Create a multi-sheet Excel file
        with pd.ExcelWriter("Pulsar_Population_Definitive.xlsx", engine='openpyxl') as writer:
            df_unfiltered.to_excel(writer, sheet_name='Without Filters', index=False)
            df_filtered.to_excel(writer, sheet_name='With Filters', index=False)
            
        print("Done! Full population saved with all parameters in 'Pulsar_Population_Definitive.xlsx'")
    else:
        print("No data generated during the entire run.")

if __name__ == "__main__":
    run_batch()