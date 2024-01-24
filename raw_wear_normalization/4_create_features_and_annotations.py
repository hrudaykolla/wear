import json
import os
import numpy as np
import pandas as pd

from utils import apply_sliding_window, label_dict, convert_labels_to_annotation_json

# define split
sbjs = [['sbj_0', 'sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5'], ['sbj_6', 'sbj_7', 'sbj_8',
                                                                 'sbj_9', 'sbj_10', 'sbj_11'], ['sbj_12', 'sbj_13', 'sbj_14', 'sbj_15', 'sbj_16', 'sbj_17']]

# change these parameters
window_overlap = 50
window_size = 50  # 25/50/100
frames = 60  # 30/60/120
stride = 30  # 15/30/60

# change output folder
raw_inertial_folder = '../data/wear/raw_clipped_scaled/inertial'
inertial_folder = '../data/wear/processed_raw_clipped_scaled/inertial_features/{}_frames_{}_stride'.format(
    frames, stride)

os.makedirs(inertial_folder, exist_ok=True)

# fixed dataset properties
nb_sbjs = 18
fps = 60
sampling_rate = 50

for i, split_sbjs in enumerate(sbjs):
    wear_annotations = {'version': 'Wear',
                        'database': {}, 'label_dict': label_dict}
    for sbj in split_sbjs:
        raw_inertial_sbj = pd.read_csv(os.path.join(
            raw_inertial_folder, sbj + '.csv'), index_col=None)
        inertial_sbj = raw_inertial_sbj.replace(
            {"label": label_dict}).fillna(-1).to_numpy()
        inertial_sbj[:, -1] += 1
        _, win_sbj, _ = apply_sliding_window(
            inertial_sbj, window_size, window_overlap)
        flipped_sbj = np.transpose(win_sbj[:, :, 1:], (0, 2, 1))
        flat_win_sbj = win_sbj.reshape(win_sbj.shape[0], -1)
        output_inertial = flipped_sbj.reshape(flipped_sbj.shape[0], -1)

        np.save(os.path.join(inertial_folder, sbj + '.npy'), output_inertial)
