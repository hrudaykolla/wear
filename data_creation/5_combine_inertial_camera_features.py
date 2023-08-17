import os
import numpy as np

sbjs = ['sbj_0', 'sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5', 'sbj_6', 'sbj_7', 'sbj_8', 'sbj_9', 'sbj_10', 'sbj_11', 'sbj_12', 'sbj_13', 'sbj_14', 'sbj_15', 'sbj_16', 'sbj_17']

inertial_folder = './data/wear/lstm_processed/inertial_features/60_frames_30_stride'
i3d_folder = './data/wear/processed/i3d_features/60_frames_30_stride'
combined_folder = './data/wear/lstm_processed/combined_features/60_frames_30_stride'

for i in range(1,4):
    temp_inertial_folder = inertial_folder+'/wear_split_'+ str(i)
    temp_combined_folder = combined_folder+'/wear_split_'+ str(i)
    for sbj in sbjs:
        inertial_features = np.load(os.path.join(temp_inertial_folder, sbj + '.npy'))
        i3d_features = np.load(os.path.join(i3d_folder, sbj + '.npy'))
        try:
            combined_features = np.concatenate((inertial_features, i3d_features), axis=1)
        except ValueError:
            print('had to chop')
            combined_features = np.concatenate((inertial_features[:i3d_features.shape[0], :], i3d_features), axis=1)
        
        # Ensure the destination folder exists, if not, create it
        os.makedirs(temp_combined_folder, exist_ok=True)
        np.save(os.path.join(temp_combined_folder, sbj + '.npy'), combined_features)
    temp_inertial_folder = inertial_folder
    temp_combined_folder = combined_folder