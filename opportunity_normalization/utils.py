from datetime import datetime, timedelta, date
from glob import glob
import os
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.io import wavfile

label_dict = { 
                'Open_Door1' : 0,
                'Open_Door2' : 1,
                'Close_Door1' : 2,
                'Close_Door2' : 3,
                'Open_Fridge' : 4,
                'Close_Fridge' : 5,
                'Open_Dishwasher' : 6,
                'Close_Dishwasher' : 7,
                'Open_Drawer1' : 8,
                'Close_Drawer1' : 9,
                'Open_Drawer2' : 10,
                'Close_Drawer2' : 11,
                'Open_Drawer3' : 12,
                'Close_Drawer3' : 13,
                'Clean_Table' : 14,
                'Drink_Cup' : 15,
                'Toggle_Switch' : 16
}

def convert_labels_to_annotation_json(labels, sampling_rate, fps, l_dict):
    annotations = []
    curr_start_i = 0
    curr_end_i = 0
    curr_label = labels[0]

    for i, l in enumerate(labels):
        if curr_label != l:
            act_start = curr_start_i / sampling_rate
            act_end = curr_end_i / sampling_rate
            act_label = curr_label
            # create annotation
            if act_label != 'null' and not pd.isnull(act_label):
                anno = {
                    'label': act_label,
                    'segment': [
                        act_start,
                        act_end
                    ],
                    'segment (frames)': [
                        act_start * fps,
                        act_end * fps
                    ],
                    'label_id': l_dict[act_label]
                    }
                annotations.append(anno)  
            curr_label = l
            curr_start_i = i + 1
            curr_end_i = i + 1
        else:
            curr_end_i += 1
    return annotations

def apply_sliding_window(data, window_size, window_overlap):
    output_x = None
    output_y = None
    output_sbj = []
    for i, subject in enumerate(np.unique(data[:, 0])):
        subject_data = data[data[:, 0] == subject]
        subject_x, subject_y = subject_data[:, :-1], subject_data[:, -1]
        tmp_x, _ = sliding_window_samples(subject_x, window_size, window_overlap)
        tmp_y, _ = sliding_window_samples(subject_y, window_size, window_overlap)

        if output_x is None:
            output_x = tmp_x
            output_y = tmp_y
            output_sbj = np.full(len(tmp_y), subject)
        else:
            output_x = np.concatenate((output_x, tmp_x), axis=0)
            output_y = np.concatenate((output_y, tmp_y), axis=0)
            output_sbj = np.concatenate((output_sbj, np.full(len(tmp_y), subject)), axis=0)

    output_y = [[i[-1]] for i in output_y]
    return output_sbj, output_x, np.array(output_y).flatten()

def sliding_window_samples(data, win_len, overlap_ratio=None):
    """
    Return a sliding window measured in seconds over a data array.

    :param data: dataframe
        Input array, can be numpy or pandas dataframe
    :param length_in_seconds: int, default: 1
        Window length as seconds
    :param sampling_rate: int, default: 50
        Sampling rate in hertz as integer value
    :param overlap_ratio: int, default: None
        Overlap is meant as percentage and should be an integer value
    :return: tuple of windows and indices
    """
    windows = []
    indices = []
    curr = 0
    overlapping_elements = 0

    if overlap_ratio is not None:
        if not ((overlap_ratio / 100) * win_len).is_integer():
            float_prec = True
        else:
            float_prec = False
        overlapping_elements = int((overlap_ratio / 100) * win_len)
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    changing_bool = True
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        if (float_prec == True) and (changing_bool == True):
            curr = curr + win_len - overlapping_elements - 1
            changing_bool = False
        else:
            curr = curr + win_len - overlapping_elements
            changing_bool = True

    return np.array(windows), np.array(indices)