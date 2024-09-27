# Monkey Dataset SfN 2024

This repository contains two Python scripts for visualizing and performing example decoding analyses on neural and kinematic data of a monkey performing a finger task, available [here]().

## Files

- `decoding_example_nwb.py`
- `visualize_data_nwb.py`

## Requirements

This code has been tested on Python 3.10. To use the code, clone the repository, create a new Python environment, and install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Visualization Tool (`visualize_data_nwb.py`)

This file provides an easy tool to visualize the finger kinematics and neural data (NWB format) from the datasets.

### Usage:
1. Run the script:
   ```bash
   python visualize_data_nwb.py
   ```
2. Click on the **Browse** button in the GUI.
3. Navigate to the folder where the NWB files are stored and select them for visualization.

## Decoding Example (`decoding_example_nwb.py`)

This file contains an example decoding analysis using ridge regression with history.

### Usage:
1. Navigate to the folder containing the NWB files.
2. Run the script:
   ```bash
   python decoding_example_nwb.py
   ```
3. Follow the prompts to:
   - Choose a training and testing day.
   - Select the corresponding target style:
     - `CO` for center-out
     - `RD` for random
   - Specify the amount of history to use as features (in number of 20ms bins).
   - Choose the type of neural feature to use:
     - `SBP` for spiking band power
     - `TCFR` for threshold crossings

## Contact Information

For any issues or questions, please reach out to Hisham Temmar at htemmar@umich.edu.