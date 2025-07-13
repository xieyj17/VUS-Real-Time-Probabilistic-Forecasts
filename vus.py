import pandas as pd 
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm 

def generate_binned_df(df, num_bins = 100):
    df['time_bin'], bin_edges = pd.qcut(df['normalized_time'], num_bins, labels=False, duplicates='drop', retbins=True)
    binned_df = df.groupby(['game_num', 'time_bin']).agg(
        avg_home_WP=('home_WP', 'mean'),
        mode_actual_result=('actual_result', lambda x: x.mode().iloc[0])
    ).reset_index()

    bin_widths = np.diff(bin_edges)
    bin_labels = np.arange(len(bin_widths))
    bin_width_map = dict(zip(bin_labels, bin_widths))

    return binned_df, bin_width_map

def calculate_roc_and_auc(binned_df, bin_index):
    # get a slice of the binned_df 
    slice_df = binned_df[binned_df["time_bin"] == bin_index]
    # Calculate the AUC for each game
    fpr, tpr, thresholds = roc_curve(slice_df['mode_actual_result'], slice_df['avg_home_WP'])
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr

def get_vus(binned_df, bin_width_map, num_bins=100):
    """
    Calculates VUS and returns both the VUS value and the list of AUCs per bin.
    """
    vus = 0.0
    auc_list = []
    for bin_index in range(num_bins):
        # Check if bin exists in the dataframe for this bootstrap sample
        if bin_index in binned_df['time_bin'].values:
            roc_auc, _, _ = calculate_roc_and_auc(binned_df, bin_index)
            if not np.isnan(roc_auc):
                vus += roc_auc * bin_width_map.get(bin_index, 0)
            # We append the roc_auc (even if NaN) to maintain a list of size num_bins
            auc_list.append(roc_auc)
        else:
            # If the bin is missing entirely in the sample, append NaN
            auc_list.append(np.nan)
            
    return vus, auc_list

def plot_vus(binned_df, bin_width_map, num_bins = 100):
    print("Generating 3D plot...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Use a colormap to make the lines distinct
    colors = plt.cm.viridis(np.linspace(0, 1, num_bins))

    # Get the unique, sorted time bins to iterate over
    time_bins = np.sort(binned_df['time_bin'].unique())

    for bin_index in time_bins:
        _, fpr, tpr = calculate_roc_and_auc(binned_df, bin_index)
        
    # Plot the ROC curve only if it was successfully calculated
        if fpr is not None and tpr is not None:
            # Create an array of the bin_index with the same shape as fpr and tpr.
            # This will be our new x-coordinate.
            x_coords = np.full_like(fpr, bin_index)
            
            try:
                # The key command: plot(x,y,z).
                # x = Time Bin, y = FPR, z = TPR
                ax.plot(x_coords, fpr, tpr, label=f'Bin {bin_index}', color=colors[bin_index])
            except ValueError as e:
                # Catch potential plotting errors and skip the problematic bin
                print(f"\n--- Could not plot Bin {bin_index} ---")
                print(f"Error: {e}")
                print("This bin will be skipped.\n")


    # Plot a semi-transparent plane representing "random chance" where TPR = FPR.
    # In our new coordinate system, this is the plane where z = y.
    print("Adding reference plane for random chance (TPR = FPR)...")
    x_range = np.linspace(0, num_bins, 2)
    y_range = np.linspace(0, 1, 2)
    x_plane, y_plane = np.meshgrid(x_range, y_range)
    ax.plot_surface(x_plane, y_plane, y_plane, alpha=0.2, color='navy') # Note: z is same as y


    # Set the labels for the new axes
    ax.set_xlabel('Time Bin', labelpad=15)
    ax.set_ylabel('False Positive Rate (FPR)', labelpad=15)
    ax.set_zlabel('True Positive Rate (TPR)', labelpad=15)
    ax.set_title('ROC Curves by Time Bin')

    # Set the limits for each axis
    ax.set_xlim(0, num_bins)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Add a color bar to indicate the progression of time bins
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_bins))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Time Bin')

    # Adjust the viewing angle for a better perspective on the new orientation
    ax.view_init(elev=25, azim=-125)

    print("Plot generated. Displaying now.")
    plt.show()

def bootstrap_vus(original_df, n_bootstraps=1000, alpha=0.05, num_bins=100):
    """
    Performs bootstrapping on the original dataset to estimate the VUS confidence interval.

    Args:
        original_df (pd.DataFrame): The original, unbinned dataframe.
        n_bootstraps (int): The number of bootstrap samples to generate.
        alpha (float): The significance level for the confidence interval.
        num_bins (int): The number of time bins to use.

    Returns:
        tuple: A tuple containing:
            - vus_distribution (list): VUS values from each bootstrap sample.
            - lower_bound (float): The lower bound of the VUS confidence interval.
            - upper_bound (float): The upper bound of the VUS confidence interval.
            - auc_estimates_df (pd.DataFrame): DataFrame of AUCs for each bin and bootstrap.
    """
    np.random.seed(42) # For reproducibility
    vus_distribution = []
    
    # Get the unique game numbers from the original data
    game_numbers = original_df['game_num'].unique()
    
    # Initialize a list of lists to store AUC values for each bin across all bootstraps
    auc_estimates_per_bin = [[] for _ in range(num_bins)]

    print(f"Starting bootstrapping with {n_bootstraps} samples...")
    
    for i in range(n_bootstraps):
        # 1. Resample with replacement by game_num to create a bootstrap sample of games
        boot_game_nums = np.random.choice(game_numbers, size=len(game_numbers), replace=True)
        
        # 2. Create the bootstrap dataframe from the original data
        boot_df = pd.concat([original_df[original_df['game_num'] == g] for g in boot_game_nums])
        
        # 3. Generate the binned dataframe for this specific bootstrap sample
        binned_boot_df, bin_width_map = generate_binned_df(boot_df, num_bins)
        
        # 4. Calculate VUS for this bootstrap sample
        boot_vus = get_vus(binned_boot_df, bin_width_map, num_bins)
        vus_distribution.append(boot_vus)
        
        # Also, store the AUC for each bin for the visualization
        for bin_index in range(num_bins):
            if bin_index in binned_boot_df['time_bin'].values:
                auc, _, _ = calculate_roc_and_auc(binned_boot_df, bin_index)
                auc_estimates_per_bin[bin_index].append(auc)
            else:
                auc_estimates_per_bin[bin_index].append(np.nan) # Handle missing bins

        # Update progress
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_bootstraps} bootstraps.")

    # Calculate the confidence interval using the percentile method
    lower_bound = np.percentile(vus_distribution, (alpha / 2) * 100)
    upper_bound = np.percentile(vus_distribution, (1 - alpha / 2) * 100)
    
    # Convert AUC estimates to a DataFrame for easier plotting
    auc_estimates_df = pd.DataFrame(auc_estimates_per_bin).transpose()
    auc_estimates_df.columns = [f'bin_{i}' for i in range(num_bins)]

    print("\nBootstrap analysis complete.")
    return vus_distribution, lower_bound, upper_bound, auc_estimates_df

def plot_auc_surface_with_confidence(auc_estimates_df, alpha=0.05):
    """
    Plots the mean AUC per time bin with confidence bands.

    Args:
        auc_estimates_df (pd.DataFrame): DataFrame of AUCs for each bin and bootstrap.
        alpha (float): The significance level for the confidence interval.
    """
    # Calculate mean and confidence intervals for AUCs in each bin
    mean_aucs = auc_estimates_df.mean()
    lower_aucs = auc_estimates_df.quantile(alpha / 2)
    upper_aucs = auc_estimates_df.quantile(1 - alpha / 2)
    
    time_bins = np.arange(len(mean_aucs))

    plt.figure(figsize=(14, 8))
    # Plot the mean AUC line
    plt.plot(time_bins, mean_aucs, label='Mean AUC', color='navy', lw=2)
    # Shade the confidence interval
    plt.fill_between(time_bins, lower_aucs, upper_aucs, color='skyblue', alpha=0.5,
                     label=f'{(1-alpha)*100}% Confidence Interval')
    
    plt.xlabel('Time Bin')
    plt.ylabel('Area Under Curve (AUC)')
    plt.title('AUC by Time Bin with Bootstrapped Confidence Intervals')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0.4, 1.0) # Adjust ylim to better see the variation
    plt.xlim(0, len(time_bins) - 1)
    plt.show()


def _bootstrap_iteration(original_df, game_numbers, num_bins):
    """
    Performs a single bootstrap iteration. This is our "worker" function.
    """
    # 1. Resample game numbers to create a bootstrap sample of games
    boot_game_nums = np.random.choice(game_numbers, size=len(game_numbers), replace=True)
    
    # 2. Create the bootstrap dataframe
    boot_df = original_df[original_df['game_num'].isin(boot_game_nums)].copy()
    
    # 3. Generate the binned dataframe for this sample
    binned_boot_df, bin_width_map = generate_binned_df(boot_df, num_bins)
    
    # 4. Calculate VUS and get the list of AUCs for this sample
    boot_vus, aucs_for_each_bin = get_vus(binned_boot_df, bin_width_map, num_bins)
    
    return boot_vus, aucs_for_each_bin


def bootstrap_vus_parallel(original_df, n_bootstraps=1000, alpha=0.05, num_bins=100):
    """
    Performs bootstrapping in parallel to estimate the VUS confidence interval.
    """
    print(f"Starting parallel bootstrapping with {n_bootstraps} samples...")
    
    game_numbers = original_df['game_num'].unique()
    
    # Use joblib to run the iterations in parallel
    # n_jobs=-1 tells joblib to use all available CPU cores
    # The 'with' block ensures the parallel pool is managed correctly
    with Parallel(n_jobs=-1) as parallel:
        results = parallel(
            delayed(_bootstrap_iteration)(original_df, game_numbers, num_bins) 
            for _ in tqdm(range(n_bootstraps), desc="Bootstrapping")
        )
    
    # Unpack the results from the list of tuples
    vus_distribution, auc_estimates_per_iteration = zip(*results)
    
    # Convert the AUC estimates to a DataFrame
    auc_estimates_df = pd.DataFrame(auc_estimates_per_iteration, columns=[f'bin_{i}' for i in range(num_bins)])
    
    # Calculate confidence interval, filtering out potential NaNs
    vus_dist_clean = [v for v in vus_distribution if not np.isnan(v)]
    lower_bound = np.percentile(vus_dist_clean, (alpha / 2) * 100)
    upper_bound = np.percentile(vus_dist_clean, (1 - alpha / 2) * 100)
    
    return vus_distribution, lower_bound, upper_bound, auc_estimates_df
