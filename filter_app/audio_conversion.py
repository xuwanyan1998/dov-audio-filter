from gccphat import gcc_phat
import pandas as pd
import numpy as np
import pickle
import time

COLNAMES = [*[f'gccphat_{i}_{j}_{d}' for i in range(4)
                                     for j in range(i+1, 4)
                                     for d in ['maxshift', 'auc', 'peakval']],
            *[f'gccphatval_{i}_{j}_{k}' for i in range(4)
                                        for j in range(i+1, 4)
                                        for k in range(23)]]

with open('model-45to45.sav', 'rb') as f:
    CLASSIFIER = pickle.load(f)

def _get_featurized_data(frame_signals, fs, max_tau):
    df = pd.DataFrame(columns=COLNAMES)
    data_row = {}
    for i in range(4):
        for j in range(i+1,4):
            gcc_phat_data = gcc_phat(frame_signals[i], frame_signals[j],
                                  fs = fs,
                                  max_tau=max_tau, interp=1)
            data_row[f'gccphat_{i}_{j}_peakval'] = gcc_phat_data[1][11]
            data_row[f'gccphat_{i}_{j}_auc'] = np.sum(gcc_phat_data[1])
            data_row[f'gccphat_{i}_{j}_maxshift'] = gcc_phat_data[0]
            # print(gcc_phat_data)

            raw_gccphat = gcc_phat_data[1]
            #np.pad(gcc_phat_data[1], ((23-len(gcc_phat_data[1]))//2,), constant_values=0)
            while len(raw_gccphat) < 23:
                raw_gccphat = np.insert(np.append(raw_gccphat, [0]), 0, 0)
                # print(f'PADDING {i} {j}')
            for k in range(23):
                data_row[f'gccphatval_{i}_{j}_{k}'] = raw_gccphat[k]

    df = df.append(data_row, ignore_index=True)
    return df.values

def process_signal(audio_signals, frame_timelen, fs, max_tau):
    frame = True

    sample_len = len(audio_signals[0])
    frame_indexlen = int(fs * frame_timelen)

    hann_window = 0.5 * 1 + np.cos(2 * np.pi * np.arange(sample_len) / frame_indexlen)
    hann_signals = [ [] for i in range(4) ]

    for i in range(4):
        hann_signals[i] = hann_window * audio_signals[i]

    cursor = 0

    full_sig = [ [] for i in range(4) ]

    iter_timestamp = None

    while cursor <= sample_len - frame_indexlen:

        cur_time = time.time()
        if iter_timestamp is not None:
            time_delta =  cur_time - iter_timestamp
            print(time_delta)
        iter_timestamp = cur_time

        frame_sig_hann = [ [] for i in range(4) ]
        frame_sig = [ [] for i in range(4) ]
        for i in range(4):
            frame_sig[i] = audio_signals[i][cursor:cursor+frame_indexlen]
            frame_sig_hann[i] = hann_signals[i][cursor:cursor+frame_indexlen]

        data_featurized = _get_featurized_data(frame_sig_hann, fs, max_tau)

        data_featurized[np.isnan(data_featurized)] = 0.0

        facing = CLASSIFIER.predict(data_featurized)
        if facing == 1:
            print(f"SEGMENT {cursor/fs} FACING!")
            pass
        else:
            frame_sig = [ [0] * frame_indexlen for i in range(4) ]

        for i in range(4):
            full_sig[i] = np.concatenate((full_sig[i], frame_sig[i]), axis=0)
        cursor += frame_indexlen

    return full_sig
