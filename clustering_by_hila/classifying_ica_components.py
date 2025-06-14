import matplotlib.image as mpimg
import mne
import os
import cv2
from picard import picard
from numpy.linalg import inv
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import griddata
import argparse


def filter_signal(data, fs):
    # apply notch filter to remove 50Hz noise
    b, a = signal.iirnotch(50, 30, fs)
    data = signal.filtfilt(b, a, data, axis=1)
    # apply bandpass filter to remove high frequency noise
    b, a = signal.butter(4, [35, 249], fs=fs, btype='band')
    filtered_signal = signal.filtfilt(b, a, data, axis=1)
    # add notch filter of 200 hz
    b, a = signal.iirnotch(200, 30, fs)
    filtered_signal = signal.filtfilt(b, a, filtered_signal)
    # add notch filter of 100 hz
    b, a = signal.iirnotch(100, 30, fs)
    filtered_signal = signal.filtfilt(b, a, filtered_signal)
    return filtered_signal


def wavelet_denoising(emg_data_input, fs, wavelet, window_size=10, level=5):
    emg_data = emg_data_input.copy()
    print("=====================")
    print(f"Denoising signal using {wavelet} Wavelet ...")
    for source in range(len(emg_data)):
        for i in range(0, len(emg_data[source]) - window_size * fs, window_size * fs):
            signal = emg_data[source, i:i + window_size * fs]
            coefficients = pywt.wavedec(signal, wavelet, level)
            for j in range(1, len(coefficients)):
                coefficients[j] = pywt.threshold(coefficients[j], np.std(coefficients[j]))
            denoised_signal = pywt.waverec(coefficients, wavelet, level)
            emg_data[source, i:i + window_size * fs] = denoised_signal
    return emg_data


def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = x - mean
    return centered, mean


def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T)) / n


def whiten(emg_signal):
    cov_matrix = covariance(emg_signal)
    U, S, V = np.linalg.svd(cov_matrix)
    d = np.diag(1.0 / np.sqrt(S))
    whiteM = np.dot(U, np.dot(d, U.T))
    emg_signal = np.dot(whiteM, emg_signal)
    return emg_signal


def calc_ica_components(emg_data, wavelet, participant_ID, session_folder_path):
    number_of_channels = emg_data.shape[0]
    K, W, Y = picard(emg_data, n_components=number_of_channels, ortho=True, extended=True, whiten=False,
                     max_iter=300)  # ICA algorithm
    np.save(
        fr"{session_folder_path}/{participant_ID}_{os.path.basename(session_folder_path)}_{wavelet}_W", W)
    np.save(
        fr"{session_folder_path}/{participant_ID}_{os.path.basename(session_folder_path)}_{wavelet}_Y", Y)
    return W


def norm(arr):
    myrange = np.nanmax(arr) - np.nanmin(arr)
    norm_arr = (arr - np.nanmin(arr)) / myrange
    return norm_arr


def plot_ica_heatmap(image_path, x_coor, y_coor, participant_ID, session_folder_path, number_of_channels, W,
                     wavelet):
    image, height, width = image_load(image_path)  # load image
    # calculations for the heatmap
    inverse = np.absolute(inv(W))
    grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]
    points = np.column_stack((x_coor, y_coor))
    f_interpolate = []
    for i in range(number_of_channels):
        interpolate_data = griddata(points, inverse[:, i], (grid_x, grid_y), method='linear')
        norm_arr = norm(interpolate_data)
        f_interpolate.append(norm_arr)  # plot heatmap
    np.save(
        fr"{session_folder_path}/{participant_ID}_{os.path.basename(session_folder_path)}_heatmap_{wavelet}",
        f_interpolate)
    fig, axs = plt.subplots(2, int(number_of_channels / 2), figsize=(16, 8), dpi=300)
    axs = axs.ravel()
    fig.subplots_adjust(hspace=0, wspace=0.01)
    for i in range(number_of_channels):
        axs[i].imshow(image)
        axs[i].pcolormesh(f_interpolate[i], cmap='jet', alpha=0.5)
        axs[i].set_title("ICA Source %d" % (i + 1))
        axs[i].axis('off')
    plt.suptitle(f"ICA components heatmaps of {participant_ID}{os.path.basename(session_folder_path)}")
    plt.savefig(
        fr"{session_folder_path}/{participant_ID}_{os.path.basename(session_folder_path)}_heatmap_{wavelet}.png")
    plt.close()

def perform_ica_algorithm(edf_path, participant_ID, session_number, session_folder_path, image_path, x_coor, y_coor,
                          number_of_channels=16, down_sample_flag=True, down_sample_fs=800):
    print("=====================")
    print(f"processing participant {participant_ID}_{session_number}")
    print("=====================")
    print(f"loading edf file ...")
    emg_file = mne.io.read_raw_edf(edf_path)
    fs = int(emg_file.info['sfreq'])  # sampling frequency
    emg_data = emg_file.get_data()[:number_of_channels]
    print("=====================")
    print(f"Filtering signal ...")
    emg_data = filter_signal(emg_data, fs)
    if down_sample_flag:
        print("=====================")
        print(f"Downsampling signal ...")
        # downsample the signal to 800 Hz
        emg_data = signal.resample(emg_data, int(emg_data.shape[1] / fs * down_sample_fs), axis=1)
        fs = down_sample_fs  # update the sampling frequency
    good_wavelets = ['db15']
    for wavelet in good_wavelets:
        start_time_wavelet = pd.Timestamp.now()
        print("=====================")
        print(f"Denoising signal ...")
        emg_data = wavelet_denoising(emg_data, fs, wavelet)
        end_time_wavelet = pd.Timestamp.now()
        print(f"wavelet denoising time: {end_time_wavelet - start_time_wavelet}")
        # print("=====================")
        print(f"Whitening signal ...")
        emg_data, mean = center(emg_data)
        # emg_data, std = standardize(emg_data)
        emg_data = whiten(emg_data)
        # print(np.round(covariance(emg_data)))
        print("=====================")
        print("running ICA algorithm")
        start_time_ica = pd.Timestamp.now()
        W = calc_ica_components(emg_data, wavelet, participant_ID, session_folder_path)
        end_time_ica = pd.Timestamp.now()
        print(f"ICA time: {end_time_ica - start_time_ica}")
        plot_ica_heatmap(image_path, x_coor, y_coor, participant_ID, session_folder_path, number_of_channels, W,
                         wavelet)
    # end time of the processing
    print(f"processing participant {participant_ID}_{session_number} done")
    print("=====================")


def image_load(image_path):
    # load the image, write the path where the image is saved (if there is no image uncomment these two lines)
    img = plt.imread(image_path)
    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = mpimg.imread(image_path)  # for heatmap

    # image dimensions
    height = img.shape[0]
    width = img.shape[1]

    return image, height, width


def plot_heatmap_classification(order_electrode, participant_folder, session_folder, session_number, w_mat, wavelet,
                                data_folder, image_path, centroids_lst, n=17):
    image, height, width = image_load(image_path)  # load image
    inverse = np.absolute(inv(w_mat))
    grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]
    points = np.column_stack((x_coor, y_coor))
    f_interpolate = []
    for i in range(16):
        f_interpolate.append(griddata(points, inverse[:, i], (grid_x, grid_y), method='linear'))

    # First plot data
    fig1, axs1 = plt.subplots(3, 8, figsize=(10, 10), dpi=300)
    axs1 = axs1.ravel()
    fig1.subplots_adjust(hspace=0, wspace=0.001)
    for i in range(len(axs1)):
        if i < len(f_interpolate):
            axs1[i].imshow(image)
            axs1[i].pcolormesh(f_interpolate[i], cmap='jet', alpha=0.5)
            axs1[i].set_title(f"Cluster {order_electrode[i] + 1}")
            axs1[i].axis('off')
        else:
            axs1[i].axis('off')
    fig1.suptitle(
        f'Participant {participant_folder}_{session_number} ICA sources heatmaps \nclassified by the Facial Muscle Atlas',
        size=20)

    # Save the first plot to a buffer
    fig1.canvas.draw()
    first_plot_image = np.frombuffer(fig1.canvas.tostring_rgb(), dtype='uint8')
    first_plot_image = first_plot_image.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig1)

    # Load image for the second plot
    image_paul, height, width = image_load(image_path)  # Assuming this function returns the image, height, and width

    # Second plot data
    fig2, axs2 = plt.subplots(3, 8, figsize=(10, 10), dpi=300)
    axs2 = axs2.ravel()
    fig2.subplots_adjust(hspace=0, wspace=0.001)
    for i in range(len(axs2)):
        if i < n:
            image = centroids_lst[i].reshape(height, width).copy()
            image[image == 0] = np.nan
            axs2[i].imshow(image_paul, cmap='gray')
            axs2[i].pcolormesh(image, cmap='jet', alpha=0.5)
            axs2[i].set_title(f"cluster {i + 1}")
            axs2[i].axis('off')
        else:
            axs2[i].axis('off')
    fig2.suptitle("The Muscle Atlas", size=20)

    # Save the second plot to a buffer
    fig2.canvas.draw()
    second_plot_image = np.frombuffer(fig2.canvas.tostring_rgb(), dtype='uint8')
    second_plot_image = second_plot_image.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig2)

    # Combine both plots into a single figure
    combined_fig, axs_combined = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
    combined_fig.subplots_adjust(wspace=0.001)

    # First plot on the left
    axs_combined[0].imshow(first_plot_image)
    axs_combined[0].axis('off')

    # Second plot on the right
    axs_combined[1].imshow(second_plot_image)
    axs_combined[1].axis('off')
    plt.savefig(
        fr'{data_folder}\{participant_folder}\{session_folder}\{participant_folder}_{session_number}_{wavelet}_ica_heatmap_classification.png')
    plt.close()


def classify_participant_components_using_atlas(participant_data, new_data_path, centroids_lst,
                                                participant_ID, session_folder_path, session_number, threshold,
                                                wavelet, image_path):
    image, height, width = image_load(image_path)
    ica_electrode_order = [0 for i in range(16)]
    flags_list = [False for i in range(17)]
    min_dist_list = [0 for i in range(17)]
    # create a list for each participant with the channels that were assigned to each cluster
    for i in range(len(participant_data)):
        dist_from_centroids = []
        for centroid in centroids_lst[:-1]:
            centroid = centroid.reshape(height, width)
            dist = np.linalg.norm(participant_data[i, :] - centroid)
            dist_from_centroids.append(dist)
        # find distance from closest centroid (out of the 16 real clusters)
        closest_centroid_dist = np.min(dist_from_centroids)
        closest_centroid_index = np.argmin(dist_from_centroids)
        if closest_centroid_dist < threshold:
            if not flags_list[closest_centroid_index]:
                flags_list[closest_centroid_index] = True
                min_dist_list[closest_centroid_index] = closest_centroid_dist
                ica_electrode_order[i] = closest_centroid_index
            else:
                if min_dist_list[closest_centroid_index] > closest_centroid_dist:
                    # get the index of the channel that was already assigned to the cluster
                    index = ica_electrode_order.index(closest_centroid_index)
                    ica_electrode_order[index] = 16
                    ica_electrode_order[i] = closest_centroid_index
                    min_dist_list[closest_centroid_index] = closest_centroid_dist
                else:
                    # if the channel was already assigned to a cluster, assign it to the garbage cluster
                    ica_electrode_order[i] = 16
        else:
            # if the channel was already assigned to a cluster, assign it to the garbage cluster
            ica_electrode_order[i] = 16
    np.save(fr'{session_folder_path}\{participant_ID}_{session_number}_{wavelet}_electrode_order.npy', ica_electrode_order)
    ica_after_order = np.zeros_like(participant_data)
    for i in range(16):
        if int(ica_electrode_order[i]) != 16:
            ica_after_order[ica_electrode_order[i], :] = participant_data[i, :]
    W = np.load(fr'{session_folder_path}\{participant_ID}_{session_number}_{wavelet}_W.npy')
    plot_heatmap_classification(ica_electrode_order, participant_folder, session_folder, session_number, W, wavelet,
                                new_data_path, image_path, centroids_lst)
    print(f"finished classifying {participant_ID}_{session_number}")

atlas_folder = "./revision_analysis/clustering_by_hila/atlas"
x_coor_path = os.path.join(atlas_folder, 'side_x_coor.npy')
y_coor_path = os.path.join(atlas_folder, 'side_y_coor.npy')
image_path = os.path.join(atlas_folder, 'side.jpg')
x_coor = np.load(x_coor_path)
y_coor = np.load(y_coor_path)

print("Loading the centroids...")
threshold = np.load(f"{atlas_folder}/threshold.npy")

# Load the centroids
centroids_lst = []
for i in range(0, 17):
    current_centroid = np.load(f"{atlas_folder}/cluster_{i + 1}.npy")
    current_centroid = np.nan_to_num(current_centroid, nan=0)
    centroids_lst.append(current_centroid)


def classify_components(w, number_of_channels=16, image_path=image_path):
    image, height, width = image_load(image_path)  # load image
    # calculations for the heatmap
    inverse = np.absolute(inv(w))
    grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]
    points = np.column_stack((x_coor, y_coor))
    
    f_interpolate = []
    for i in range(number_of_channels):
        interpolate_data = griddata(points, inverse[:, i], (grid_x, grid_y), method='linear')
        norm_arr = norm(interpolate_data)
        f_interpolate.append(norm_arr)  # plot heatmap

    participant_data = np.array(f_interpolate)
    participant_data = np.nan_to_num(participant_data, nan=0)

    ica_electrode_order = [0 for i in range(16)]
    flags_list = [False for i in range(17)]
    min_dist_list = [0 for i in range(17)]
    # create a list for each participant with the channels that were assigned to each cluster
    for i in range(len(participant_data)):
        dist_from_centroids = []
        for centroid in centroids_lst[:-1]:
            centroid = centroid.reshape(height, width)
            dist = np.linalg.norm(participant_data[i, :] - centroid)
            dist_from_centroids.append(dist)
        # find distance from closest centroid (out of the 16 real clusters)
        closest_centroid_dist = np.min(dist_from_centroids)
        closest_centroid_index = np.argmin(dist_from_centroids)
        if closest_centroid_dist < threshold:
            if not flags_list[closest_centroid_index]:
                flags_list[closest_centroid_index] = True
                min_dist_list[closest_centroid_index] = closest_centroid_dist
                ica_electrode_order[i] = closest_centroid_index
            else:
                if min_dist_list[closest_centroid_index] > closest_centroid_dist:
                    # get the index of the channel that was already assigned to the cluster
                    index = ica_electrode_order.index(closest_centroid_index)
                    ica_electrode_order[index] = 16
                    ica_electrode_order[i] = closest_centroid_index
                    min_dist_list[closest_centroid_index] = closest_centroid_dist
                else:
                    # if the channel was already assigned to a cluster, assign it to the garbage cluster
                    ica_electrode_order[i] = 16
        else:
            # if the channel was already assigned to a cluster, assign it to the garbage cluster
            ica_electrode_order[i] = 16

    return ica_electrode_order

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process EMG data with ICA and classify components using atlas.')

    # Add arguments
    parser.add_argument('--project_folder', type=str, help='Path to the project root folder')
    parser.add_argument('--down_sample_fs', type=int, default=800, help='Downsampling frequency (default: 800)')
    parser.add_argument('--number_of_channels', type=int, default=16, help='Number of channels (default: 16)')
    parser.add_argument('--down_sample_flag', action='store_true', help='Enable downsampling')
    parser.add_argument('--wavelet', type=str, default='db15', help='Wavelet type (default: db15)')
    parser.add_argument('--n_clusters', type=int, default=17, help='Number of clusters (default: 17)')

    # Parse arguments
    args = parser.parse_args()
    atlas_folder = os.path.join(args.project_folder, 'atlas')
    # Load coordinates


    # Process data
    data_path = os.path.join(args.project_folder, 'data')
    for participant_folder in os.listdir(data_path):
        participant_ID = participant_folder
        participant_folder_path = os.path.join(data_path, participant_folder)

        for session_folder in os.listdir(participant_folder_path):
            session_folder_path = os.path.join(participant_folder_path, session_folder)
            edf_files_lst = []

            for file in os.listdir(session_folder_path):
                if file.endswith('.edf'):
                    edf_files_lst.append(os.path.join(session_folder_path, file))
                else:
                    continue

            for edf_file_path in edf_files_lst:
                participant_ID = participant_folder
                session_number = session_folder

                if f'{participant_ID}_{session_number}_heatmap_{args.wavelet}.npy' not in os.listdir(
                        session_folder_path):
                    perform_ica_algorithm(
                        edf_file_path,
                        participant_ID,
                        session_number,
                        session_folder_path,
                        image_path,
                        x_coor,
                        y_coor,
                        args.number_of_channels,
                        args.down_sample_flag,
                        args.down_sample_fs
                    )

                participant_data = np.load(
                    os.path.join(session_folder_path, f'{participant_ID}_{session_number}_heatmap_{args.wavelet}.npy')
                )
                participant_data = np.nan_to_num(participant_data, nan=0)

                classify_participant_components_using_atlas(
                    participant_data,
                    data_path,
                    centroids_lst,
                    participant_ID,
                    session_folder_path,
                    session_number,
                    threshold,
                    args.wavelet,
                    image_path)