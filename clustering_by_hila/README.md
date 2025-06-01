# EMG Data Processing and Analysis Pipeline

This project implements a pipeline for processing and analyzing EMG (Electromyography) data using Independent Component Analysis (ICA) and wavelet denoising, based on the analysis methodology described in [CITATION PENDING]. The pipeline includes signal filtering, component classification, and visualization through heatmaps.
![EMG Analysis Pipeline Output](Figure_Atlas.png)

## Project Structure

The project requires the following folder structure:
```
project_root/
│
├── classifying_ica_components.py
├── atlas/
│   ├── cluster_1.npy
│   ├── cluster_2.npy
│   ├── ...
│   ├── cluster_17.npy
│   ├── threshold.npy
│   ├── side_x_coor.npy
│   ├── side_y_coor.npy
│   └── side.jpg
├── data/
│   ├── participant_1/
│   │   ├── session_1/
│   │   │   └── *.edf files
│   │   └── session_2/
│   │       └── *.edf files
│   └── participant_2/
│       └── ...
│
└── requirements.txt
```

### Prerequisites
### Installation
To install the required Python packages, run:
```bash
pip install -r requirements.txt
```

This will install all necessary dependencies for running the pipeline. 
### Required Files

1. **Atlas Files** (in `atlas/` folder):
   - Cluster files (`cluster_1.npy` through `cluster_17.npy`)
   - Coordinate files (`side_x_coor.npy`, `side_y_coor.npy`)
   - Threshold file (`threshold.npy`)
   - Image File - Side view face image (`side.jpg`) for heatmap visualization

2. **Data Files**:
   - EDF files containing EMG recordings, organized by participant and session

## Running the Code

### Basic Usage

First, navigate to the project root directory:
```bash
cd /path/to/project_root
```

Then run the main script with the required command line arguments:


```bash
python classifying_ica_components.py --project_folder /path/to/project_root
```

### All Available Options

```bash
python classifying_ica_components.py --project_folder /path/to/project_root \
                     --down_sample_fs 800 \
                     --number_of_channels 16 \
                     --down_sample_flag \
                     --wavelet db15 \
                     --n_clusters 17
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--project_folder` | Path to the project root directory | Required |
| `--atlas_folder` | Path to the atlas folder containing centroids and coordinates | Required |
| `--down_sample_fs` | Downsampling frequency | 800 |
| `--number_of_channels` | Number of EMG channels | 16 |
| `--down_sample_flag` | Enable downsampling | False |
| `--wavelet` | Wavelet type for denoising | 'db15' |
| `--n_clusters` | Number of clusters for classification | 17 |

## Pipeline Steps

1. **Data Loading**: 
   - Reads EDF files containing EMG recordings
   - Processes data for each participant and session

2. **Signal Processing**:
   - Applies notch filters (50Hz, 100Hz, 200Hz)
   - Performs bandpass filtering (35-249Hz)
   - Optional downsampling
   - Wavelet denoising

3. **ICA Analysis**:
   - Signal whitening
   - ICA component calculation
   - Component visualization through heatmaps

4. **Classification**:
   - Classifies ICA components using pre-defined atlas
   - Generates visualization comparing components to atlas

## Output Files

For each processed session, the following files are generated in the session folder:

- `{participant_id}_{session_number}_{wavelet}_W.npy`: ICA mixing matrix
- `{participant_id}_{session_number}_{wavelet}_Y.npy`: ICA sources
- `{participant_id}_{session_number}_{wavelet}_heatmap.npy`: Heatmap data
- `{participant_id}_{session_number}_{wavelet}_heatmap.png`: Heatmap visualization
- `{participant_id}_{session_number}_{wavelet}_electrode_order.npy`: Classification results
- `{participant_id}_{session_number}_{wavelet}_ica_heatmap_classification.png`: Classification visualization

## Notes

- The code expects EDF files to be organized in a specific folder structure (participant/session/edf_files)
- Make sure all required atlas files are present in the atlas folder before running
- The face image (side.jpg) is required for visualization
- Ensure sufficient disk space for output files, especially with large datasets

## Troubleshooting

1. **File Not Found Errors**:
   - Verify the folder structure matches the expected format
   - Check that all required atlas files are present
   - Ensure paths are correctly specified in command line arguments

2. **Memory Issues**:
   - Consider enabling downsampling for large datasets
   - Process fewer sessions at a time if needed

3. **Processing Time**:
   - ICA and wavelet processing can be time-consuming for large datasets
   - Progress is printed to console during processing
