import torch
import numpy as np
import os

def generate_fake_dataset(num_samples=500, sequence_len=186, num_classes=5):
    """
    Generate a fake dataset in the expected format:
    {
        'samples': np.ndarray of shape [num_samples, sequence_len, 1]
        'labels': np.ndarray of shape [num_samples]
    }
    """
    # Random float ECG-like sequences
    samples = np.random.randn(num_samples, sequence_len, 1).astype(np.float32)

    # Random labels from 0 to 4
    # labels = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int64)
    # Make sure each class appears at least once
    base_labels = np.arange(num_classes)
    remaining = num_samples - num_classes
    random_labels = np.random.randint(0, num_classes, size=remaining)
    labels = np.concatenate([base_labels, random_labels])
    np.random.shuffle(labels)

    labels = labels.astype(np.int64)

    return {'samples': samples, 'labels': labels}

def save_fake_mit_data(root_dir='data', split_sizes=(500, 100, 100)):
    """
    Save fake MIT-BIH dataset to ./data/mit/{train.pt, val.pt, test.pt}
    """
    dataset_dir = os.path.join(root_dir, 'mit')
    os.makedirs(dataset_dir, exist_ok=True)

    for split_name, n_samples in zip(['train', 'val', 'test'], split_sizes):
        data = generate_fake_dataset(num_samples=n_samples)
        file_path = os.path.join(dataset_dir, f"{split_name}.pt")
        torch.save(data, file_path)
        print(f"Saved {split_name}.pt to {file_path}")

    print("\n Fake MIT-BIH dataset successfully created!")

if __name__ == '__main__':
    save_fake_mit_data()

