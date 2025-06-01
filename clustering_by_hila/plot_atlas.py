import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
from numpy.linalg import inv
import os
from tqdm import tqdm
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from skimage.transform import resize
from pathlib import Path
from collections import defaultdict


def image_load(image_path):
    """Load and process image for heatmap overlay"""
    img = plt.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = mpimg.imread(image_path)
    height = img.shape[0]
    width = img.shape[1]
    return image, height, width


def norm(data):
    """Normalize data to 0-1 range"""
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def generate_participant_heatmaps(W_matrix, x_coor, y_coor, height, width, number_of_channels):
    """Generate heatmaps from W matrix using electrode coordinates"""
    inverse = np.absolute(inv(W_matrix))
    grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]
    points = np.column_stack((x_coor, y_coor))

    participant_heatmaps = []
    for i in range(number_of_channels):
        interpolate_data = griddata(points, inverse[:, i], (grid_x, grid_y), method='linear')
        norm_arr = norm(interpolate_data)
        participant_heatmaps.append(norm_arr)

    return np.array(participant_heatmaps)


def load_electrode_order(file_path):
    """Load electrode order classification from text file"""
    electrode_order = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    electrode_order.append(int(line))
        return np.array(electrode_order)
    except Exception as e:
        print(f"Error loading electrode order from {file_path}: {e}")
        return None


def collect_participant_data(w_matrices_folder, electrode_order_folder, x_coor, y_coor, image_path):
    """Collect W matrices and electrode orders for all participants"""
    participant_data = {}
    image, height, width = image_load(image_path)

    # Get all W matrix files
    w_files = list(Path(w_matrices_folder).glob('*.npy'))

    for w_file in w_files:
        # Extract the base name (without .npy extension)
        base_name = w_file.stem

        # Look for corresponding electrode order file
        electrode_file = Path(electrode_order_folder) / f"{base_name}_comp_to_cluster.txt"

        if electrode_file.exists():
            try:
                # Load W matrix
                W_matrix = np.load(w_file)

                # Load electrode order
                electrode_order = load_electrode_order(electrode_file)

                if electrode_order is not None:
                    # Generate heatmaps
                    heatmaps = generate_participant_heatmaps(
                        W_matrix, x_coor, y_coor, height, width, W_matrix.shape[1]
                    )

                    participant_data[base_name] = {
                        'heatmaps': heatmaps,
                        'electrode_order': electrode_order,
                        'W_matrix': W_matrix
                    }
                    print(f"Loaded data for {base_name}")
            except Exception as e:
                print(f"Error processing {base_name}: {e}")
        else:
            print(f"No electrode order file found for {base_name}")

    return participant_data


def compute_cluster_means(participant_data, n_clusters=16):
    """Compute mean heatmaps for each cluster based on participant classifications"""
    cluster_means = {}
    cluster_counts = defaultdict(int)
    cluster_participants = defaultdict(list)

    # Initialize cluster accumulations
    for cluster_id in range(n_clusters):
        cluster_means[cluster_id] = []

    # Accumulate heatmaps by cluster
    for participant_id, data in participant_data.items():
        heatmaps = data['heatmaps']
        electrode_order = data['electrode_order']

        # Group heatmaps by cluster assignment
        for component_idx, cluster_id in enumerate(electrode_order):
            if 0 <= cluster_id < n_clusters:  # Valid cluster range
                cluster_means[cluster_id].append(heatmaps[component_idx])
                cluster_participants[cluster_id].append(participant_id)

    # Compute means for each cluster
    final_cluster_means = {}
    for cluster_id in range(n_clusters):
        if cluster_means[cluster_id]:
            final_cluster_means[cluster_id] = np.mean(cluster_means[cluster_id], axis=0)
            cluster_counts[cluster_id] = len(cluster_means[cluster_id])
            print(
                f"Cluster {cluster_id + 1}: {cluster_counts[cluster_id]} components from {len(set(cluster_participants[cluster_id]))} participants")
        else:
            # Create empty heatmap if no components assigned
            height, width = list(participant_data.values())[0]['heatmaps'].shape[1:]
            final_cluster_means[cluster_id] = np.zeros((height, width))
            cluster_counts[cluster_id] = 0
            print(f"Cluster {cluster_id + 1}: No components assigned")

    return final_cluster_means, cluster_counts, cluster_participants


def plot_participant_atlas(image_path, cluster_means, cluster_counts, cluster_participants, output_path):
    """Plot atlas showing mean heatmaps for each cluster"""
    image, height, width = image_load(image_path)
    n_clusters = 16

    # Calculate layout - 2 rows of 8 columns
    n_rows = 2
    n_cols = 8

    # Create figure with adjusted size for 2x8 layout
    fig = plt.figure(figsize=(30, 10), dpi=300)
    # Use negative wspace to reduce spacing between columns even more
    main_gs = fig.add_gridspec(n_rows, n_cols, hspace=0.001, wspace=-0.01)

    def crop_face(image, mask):
        """Crop face area from image"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return image, mask
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding
        padding_r = 40
        padding_c = 40
        rmin = max(rmin, 0)
        rmax = min(rmax + padding_r, image.shape[0])
        cmin = max(cmin - padding_c - 22, 0)
        cmax = min(cmax + padding_c, image.shape[1])

        return image[rmin:rmax, cmin:cmax], mask[rmin:rmax, cmin:cmax]

    for i in range(n_clusters):
        row = i // n_cols
        col = i % n_cols

        ax = fig.add_subplot(main_gs[row, col])

        # Get cluster mean heatmap
        cluster_heatmap = cluster_means[i]

        # Crop face area
        cropped_image, cropped_heatmap = crop_face(image, cluster_heatmap)

        # Resize for display
        new_height = 200
        aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
        new_width = int(new_height * aspect_ratio)
        cropped_image_resized = resize(cropped_image, (new_height, new_width), anti_aliasing=True)
        cropped_heatmap_resized = resize(cropped_heatmap, (new_height, new_width), anti_aliasing=True)

        # Display image and heatmap
        ax.imshow(cropped_image_resized, cmap='gray')

        # Mask zero values
        cropped_heatmap_resized[cropped_heatmap_resized == 0] = np.nan
        ax.pcolormesh(cropped_heatmap_resized, cmap='jet', alpha=0.5)

        # Add cluster information with larger fonts - 2 rows only
        n_participants = len(set(cluster_participants[i])) if i in cluster_participants else 0
        ax.text(0.05, 0.95, f"Cluster {i + 1}\nN = {cluster_counts[i]}",
                fontsize=22, fontweight='bold', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

        ax.axis('off')

    # Save plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Atlas saved to: {output_path}")

def main():
    """Main function to generate participant-based atlas"""
    # Configuration
    n_clusters = 16

    # Paths - adjust these to match your directory structure
    project_folder = os.getcwd()

    w_matrices_folder = os.path.join(project_folder, 'W_matrices')
    electrode_order_folder = os.path.join(project_folder, 'electrode_order')

    # Image and coordinate paths - adjust these as needed
    image_path = fr"{project_folder}\Pic1.jpg"

    # Load electrode coordinates - adjust these paths as needed
    # You might need to use generic coordinates or specify the correct scheme
    x_coor_path = fr"{project_folder}\Pic1x_coor.npy"
    y_coor_path = fr"{project_folder}\Pic1y_coor.npy"

    x_coor = np.load(x_coor_path)
    y_coor = np.load(y_coor_path)

    print("Loading participant data...")
    participant_data = collect_participant_data(
        w_matrices_folder, electrode_order_folder, x_coor, y_coor, image_path
    )

    if not participant_data:
        print("No participant data loaded. Check file paths and naming conventions.")
        return

    print(f"Loaded data for {len(participant_data)} participants")

    print("Computing cluster means...")
    cluster_means, cluster_counts, cluster_participants = compute_cluster_means(participant_data, n_clusters)

    # Create output directory
    output_dir = os.path.join(project_folder, 'plots', 'participant_atlas')
    os.makedirs(output_dir, exist_ok=True)

    # Generate atlas plot
    output_path = os.path.join(output_dir, f'Participant_Atlas.png')
    print("Generating atlas plot...")
    plot_participant_atlas(image_path, cluster_means, cluster_counts, cluster_participants, output_path)

    # Save cluster means as numpy arrays
    for cluster_id, cluster_mean in cluster_means.items():
        np.save(os.path.join(output_dir, f'cluster_{cluster_id + 1}_mean_heatmap.npy'), cluster_mean)

    print("Atlas generation complete!")


if __name__ == '__main__':
    main()