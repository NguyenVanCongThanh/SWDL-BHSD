# SETUP AND RESUME TRAINING GUIDE ON PERSONAL COMPUTER

This document provides detailed instructions on how to set up the environment, prepare the directory structure, and resume training the model on a local Linux personal computer using checkpoints from Google Colab.

---

## 1. Standard Directory Structure
To ensure the source code runs correctly, set up the project directory structure as follows:

```text
SWDL/                           # Project root directory
├── code/                       # Directory containing original source code
│   ├── BHSD2024/
│   │   ├── train_SWDL.py       # Main script to run training
│   │   ├── test_3D_util.py
│   │   └── ...
│   ├── networks/
│   │   └── ...
│   └── utils/
│       └── ...
├── data/                       # Directory containing input data
│   └── BHSD_Dataset_RemoveSkull_resampled/
│       ├── dataSet/            # PLACE TO STORE .h5 DATA FILES (e.g., ID_1653b22b_ID_c35cd0de73.h5)
│       └── fold_1/
│           └── test.list       # List of test cases (each line contains a .h5 file name)
└── model/                      # Automatically created directory to save results
    └── BHSD/
        └── SWDL_05p_fold_1/
```

### Steps to Prepare Files:
1. **Source Code:** Ensure the `code` directory is in the `SWDL` root directory.
2. **Data:**
   - Copy the original `.h5` medical imaging files into the `SWDL/data/BHSD_Dataset_RemoveSkull_resampled/dataSet/` directory.
   - Copy the `test.list` file into the `SWDL/data/BHSD_Dataset_RemoveSkull_resampled/fold_1/` directory.

---

## 2. Environment and Library Installation Steps

You can set up the complete virtual environment and all required libraries directly using the provided `environment.yml` file:

```bash
# Create the virtual environment named "swdl" from the environment.yml file
conda env create -f environment.yml

# Activate the virtual environment
conda activate swdl
```

---

## 3. Resuming Training on Local Machine from Google Colab Checkpoint

If you previously trained the model on Google Colab (using the notebook `train_colab.ipynb`) and want to continue training on your local machine, follow these steps:

### Step 3.1: Download Checkpoint and Logs from Google Drive
1. Go to your Google Drive and navigate to:
   `/content/drive/MyDrive/SWDL/model/BHSD/`
2. Download the entire folder containing your training progress (typically named `SWDL_05p_fold_1/`). This folder contains `latest_checkpoint.pth`, `log.txt`, and other intermediate checkpoints.

### Step 3.2: Place the Files in Your Local Directory
On your local machine, place the downloaded folder inside the `model/BHSD/` directory. Your local directory structure should look like this:
```text
SWDL/
├── code/
│   └── BHSD2024/
│       └── train_SWDL.py
├── data/
│   └── BHSD_Dataset_RemoveSkull_resampled/
└── model/
    └── BHSD/
        └── SWDL_05p_fold_1/              <-- PLACED HERE
            ├── latest_checkpoint.pth     <-- Checkpoint file for resuming
            ├── log.txt
            └── ...
```

### Step 3.3: Run the Local Resume Command
1. Open your terminal, navigate to the `code/BHSD2024/` directory:
   ```bash
   cd path_to_directory/SWDL/code/BHSD2024
   ```
2. Activate your virtual environment:
   ```bash
   conda activate swdl
   ```
3. Run the training script with the `--resume` flag and match the settings used in Google Colab (e.g., `--labeled_num 153`):
   ```bash
   python train_SWDL.py --labeled_num 153 --resume
   ```
   *The script will automatically detect `latest_checkpoint.pth` under `SWDL/model/BHSD/SWDL_05p_fold_1/` and continue training from the last saved iteration.*

---

## 4. Monitoring Training Progress with TensorBoard

You can monitor training metrics (Loss, Validation Dice Score, HD95, etc.) in real-time using TensorBoard:

1. Open a new terminal window and navigate to the project root directory `SWDL`.
2. Activate the virtual environment:
   ```bash
   conda activate swdl
   ```
3. Run the TensorBoard server pointing to the model directory:
   ```bash
   tensorboard --logdir model/BHSD/
   ```
4. Open your web browser and go to:
   [http://localhost:6006](http://localhost:6006)

