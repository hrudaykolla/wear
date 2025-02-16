import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns

from utils import ANETdetection, convert_segments_to_samples

# postprocessing parameters
path_to_preds = ['./results/opportunity/tridet/inertial_norm_lstm/']
seeds = [1, 2]
score_thres = [0.0, 0.2]
# 0.0, 0.05, 0.1, 0.15, 0.2, 0.25
sampling_rate = 30
json_files = [
    'data/opportunity_adl_data/annotations/opportunity_split_1.json',
    'data/opportunity_adl_data/annotations/opportunity_split_2.json',
    'data/opportunity_adl_data/annotations/opportunity_split_3.json'
]

for path in path_to_preds:
    print("Data Loading....")
    for f in score_thres:
        all_mAP = np.zeros((len(seeds), 5))
        all_recall = np.zeros((len(seeds), 18))
        all_prec = np.zeros((len(seeds), 18))
        all_f1 = np.zeros((len(seeds), 18))
        for s_pos, seed in enumerate(seeds):
            all_preds = np.array([])
            all_gt = np.array([])

            for i, j in enumerate(json_files):
                with open(j) as fi:
                    file = json.load(fi)
                anno_file = file['database']
                labels = ['null'] + list(file['label_dict'])
                label_dict = dict(zip(labels, list(range(len(labels)))))
                val_sbjs = [x for x in anno_file if anno_file[x]
                            ['subset'] == 'Validation']

                v_data = np.empty((0, 113 + 2))
                v_seg = pd.read_csv(os.path.join(
                    path, 'seed_' + str(seed), 'unprocessed_results/v_seg_opportunity_split_{}.csv'.format(int(i) + 1, seed)), index_col=None)
                for sbj in val_sbjs:
                    data = pd.read_csv(os.path.join('data/opportunity_adl_data/raw_clipped_scaled', sbj + '.csv'),
                                       index_col=False).replace({"label": label_dict}).fillna(0).to_numpy()
                    v_data = np.append(v_data, data, axis=0)

                print("Converting to Samples....")
                v_seg = v_seg.rename(
                    columns={"video_id": "video-id", "t_start": "t-start", "t_end": "t-end"})
                preds, gt, _ = convert_segments_to_samples(
                    v_seg, v_data, sampling_rate, threshold=f)
                all_preds = np.concatenate((all_preds, preds))
                all_gt = np.concatenate((all_gt, gt))

                v_seg = v_seg[v_seg.score > f]
                det_eval = ANETdetection(j, 'validation', tiou_thresholds=[
                                         0.3, 0.4, 0.5, 0.6, 0.7])

                print("Evaluating {}....".format(j))
                v_mAP, _ = det_eval.evaluate(v_seg)
                v_prec = precision_score(gt, preds, average=None)
                v_rec = recall_score(gt, preds, average=None)
                v_f1 = f1_score(gt, preds, average=None)

                all_prec[s_pos, :] += v_prec
                all_recall[s_pos, :] += v_rec
                all_f1[s_pos, :] += v_f1
                all_mAP[s_pos, :] += v_mAP
            if seed == 1:
                comb_conf = confusion_matrix(
                    all_gt, all_preds, normalize='true')
                comb_conf = np.around(comb_conf, 2)
                comb_conf[comb_conf == 0] = np.nan

                _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
                sns.heatmap(comb_conf, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens, cbar=False, annot_kws={
                    'fontsize': 16,
                })
                pred_name = path.split('/')[-2]
                _.savefig(pred_name + ".pdf")
                np.save(pred_name, all_preds)

        print("Prediction for {} with threshold {}:".format(path_to_preds, f))
        print("Individual mAP:")
        print(np.around(np.mean(all_mAP, axis=0) / len(json_files), 4) * 100)

        print("Average mAP:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_mAP) / len(json_files)
              * 100, np.std(np.mean(all_mAP, axis=1) / len(json_files)) * 100))

        print("Individual Precision:")
        print(np.around(np.mean(all_prec, axis=0) / len(json_files), 4) * 100)

        print("Average Precision:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_prec) / len(json_files)
              * 100, np.std(np.mean(all_prec, axis=1) / len(json_files)) * 100))

        print("Individual Recall:")
        print(np.around(np.mean(all_recall, axis=0) / len(json_files), 4) * 100)

        print("Average Recall:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_recall) / len(json_files)
              * 100, np.std(np.mean(all_recall, axis=1) / len(json_files)) * 100))

        print("Individual F1:")
        print(np.around(np.mean(all_f1, axis=0) / len(json_files), 4) * 100)

        print("Average F1:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_f1) / len(json_files)
              * 100, np.std(np.mean(all_f1, axis=1) / len(json_files)) * 100))
