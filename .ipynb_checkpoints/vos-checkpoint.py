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
        fpr, tpr, roc_auc = calculate_roc_and_auc(binned_df, bin_index)
        
        # Plot the ROC curve only if it was successfully calculated
        if fpr is not None and tpr is not None:
            # The key command: plot(x,y,z). Here, z is the constant bin_index.
            ax.plot(fpr, tpr, zs=bin_index, zdir='z', label=f'Bin {bin_index}', color=colors[bin_index])

    # Plot the "line of no-discrimination" (random chance) on the base plane for reference
    ax.plot([0, 1], [0, 1], zs=0, zdir='z', color='navy', linestyle='--', label='Random Chance')

    # --- 5. Style the Plot ---
    ax.set_xlabel('False Positive Rate (FPR)', labelpad=10)
    ax.set_ylabel('True Positive Rate (TPR)', labelpad=10)
    ax.set_zlabel('Time Bin', labelpad=10)
    ax.set_title('ROC Curves for Each Time Bin')

    # Set the limits for each axis
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, num_bins)

    # Add a color bar to indicate the progression of time bins
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_bins))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Time Bin')


    # Adjust the viewing angle for better perspective
    ax.view_init(elev=25, azim=-65)

    print("Plot generated. Displaying now.")
    plt.show()


