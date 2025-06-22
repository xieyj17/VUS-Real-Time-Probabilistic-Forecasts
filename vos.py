import pandas as pd 
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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

def get_vos(binned_df, bin_width_map, num_bins = 100):
    auc_list = []
    for bin_index in range(num_bins):
        auc, _, _ = calculate_roc_and_auc(binned_df, bin_index)
        auc_list.append(auc)
    
    vos = 0
    for i, auc in enumerate(auc_list):
        vos += auc * bin_width_map[i]
    return vos

def plot_vos(binned_df, bin_width_map, num_bins = 100):
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

    # --- 5. Style the Plot ---

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