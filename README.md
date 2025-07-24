# TC-SSFN
 A Cycle-consistent Temporal-Spectral Fusion Network (CTSFN) that jointly modeled local spatial-spectral structures and long-term temporal dynamics



# Time-Series Remote Sensing Dataset

This repository provides preprocessed time-series remote sensing data, including `data`, `date`, and labels for 6 different scenes.
Access from Google Drive
Link：https://drive.google.com/drive/folders/1REgFXcBW8ZTiWwbcQEKzIgNRosX2qG0C?usp=drive_link
## Data Structure

### 1. `data`
- **Format**: `[C, H, W, T]`
  - **C**: Number of spectral channels
  - **H**: Image height
  - **W**: Image width
  - **T**: Number of time steps
- **Description**: `data` stores spectral reflectance values of time-series remote sensing images, ordered by time.

### 2. `date`
- **Format**: `[T, 3]`
  - **T**: Time steps corresponding to `data`
  - **3**: `[Year, Month, Day]`
- **Description**: `date` provides the calendar date for each time step.

### 3. `label`
- **Number of Scenes**: 6 (`Scene1` ~ `Scene6`)
- **Description**: Each scene includes ground-truth labels for land-cover change, used for training and evaluating change detection models.

## Directory Structure Example
dataset/
├── data/
│ ├── scene1_data.npy
│ ├── scene2_data.npy
│ └── ...
├── date/
│ ├── scene1_date.npy
│ ├── scene2_date.npy
│ └── ...
└── label/
├── scene1_label.npy
├── scene2_label.npy
└── ...


## Usage
```python
import numpy as np

# Load data
data = np.load('dataset/data/scene1_data.npy')    # shape: [C, H, W, T]
date = np.load('dataset/date/scene1_date.npy')    # shape: [T, 3]
label = np.load('dataset/label/scene1_label.npy') # shape: [H, W]
