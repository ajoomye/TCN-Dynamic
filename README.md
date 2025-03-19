# Codes and Extracted data for Android malware detection using TCN and the CIC-Maldroid20 dataset

These are the codes for the paper following paper:

`A. Joomye, M. H. Ling, M. B. Jasser, A. M. Ramly and K. -L. A. Yau, "An Effective Temporal Convolutional Networks-based Method for Detecting Android Malware using Dynamic Extracted Features," in IEEE Access, doi: 10.1109/ACCESS.2025.3552070.`


To run training:
1. Unzip and copy the files from the `Extracted_data` folder to the same folder as the `training.py`.
2. Install the following Python dependencies using
   `pip install -r requirements.txt`
3. Run `training.py`

The weights of the published results can be found in the `final_model_weights.h5` file and the results per training in `final_model_training.csv`.
