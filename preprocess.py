import os
import numpy as np
import torch
import wfdb
from sklearn.preprocessing import StandardScaler

# AAMI class mapping
aami_mapping = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'Q': 4, '|': 4, 'f': 4, 'x': 4, 'U': 4, 'Q': 4, '~': 4
}

def extract_beats(record_name, data_dir, window_size=186): # window size = 186 (186/360 = 0.517s)
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)         # read ECG signal 
    annotation = wfdb.rdann(record_path, 'atr') # read R-peak location & beat type  

    signal = record.p_signal[:, 0]  # use channel 0 (MLII - Lead II)
    beats = []
    labels = []

    half_window = window_size // 2
    for idx, sym in zip(annotation.sample, annotation.symbol):
        if sym not in aami_mapping:
            continue

        if idx - half_window < 0 or idx + half_window > len(signal):
            continue  # skip incomplete windows

        beat = signal[idx - half_window:idx + half_window] # extract 186-sample beatw around R-peak
        if len(beat) == window_size:
            beats.append(beat)
            labels.append(aami_mapping[sym])  # map ECG heartbeat types to AAMI classes

    return np.array(beats), np.array(labels)

def preprocess_mitdb(data_dir='mitdb', output_dir='data/mit'):
    os.makedirs(output_dir, exist_ok=True)
    all_beats = []
    all_labels = []

    for record in os.listdir(data_dir):
        if record.endswith('.dat'):
            rec_id = record[:-4]
            print(f"Processing {rec_id}")
            beats, labels = extract_beats(rec_id, data_dir)
            all_beats.append(beats)
            all_labels.append(labels)

    X = np.concatenate(all_beats, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)

    # Normalize each beat independently
    scaler = StandardScaler()
    X = np.array([scaler.fit_transform(b.reshape(-1, 1)).flatten() for b in X])

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split
    train_end = int(0.7 * len(X))
    val_end = int(0.85 * len(X))

    splits = {
        'train': {'samples': X[:train_end], 'labels': y[:train_end]},
        'val': {'samples': X[train_end:val_end], 'labels': y[train_end:val_end]},
        'test': {'samples': X[val_end:], 'labels': y[val_end:]}
    }

    for name, data in splits.items():
        torch.save(data, os.path.join(output_dir, f"{name}.pt"))
        print(f"Saved {name}.pt with shape {data['samples'].shape}")

if __name__ == "__main__":
    preprocess_mitdb()