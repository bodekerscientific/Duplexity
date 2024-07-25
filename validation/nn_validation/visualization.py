import matplotlib.pyplot as plt

def plot_2d_array(data, title="2D Array Plot"):
    plt.imshow(data, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()




def metric_map_plot(metrics_result, metric, cmap="viridis", vmin=0, vmax=10, plot_path=None):
    """
    Plot metrics in a map.
    
    Parameters:
    metrics_result (dict): A dictionary with metric names as keys and metric arrays as values.
    metric (str): Name of the metric to plot.
    cmap (str): Colormap to use.
    vmin (float): Minimum value for the colormap.
    vmax (float): Maximum value for the colormap.
    plot_path (str): Path to save the plot.
    
    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(metrics_result[metric], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(metric)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(metric)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if plot_path is provided
    if plot_path:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plot_name = f"{metric}_map.png"
        plt.savefig(os.path.join(plot_path, plot_name))
    else:
        plt.show()
    
    plt.close()



