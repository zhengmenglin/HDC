
import sys
import os

# Add torchhd to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchhd'))

import torch
import torch.nn as nn
import torch.utils.data as data
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
WINDOW = 32
N_GRAM_SIZE = 4
BATCH_SIZE = 1
DIMENSIONS = 10000
NUM_LEVELS = 255
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Class
class RuleBasedDataset(data.Dataset):
    def __init__(self, csv_path, window_size=32, is_train=True, split_ratio=0.7):
        self.window_size = window_size
        
        # Load Data
        df = pd.read_csv(csv_path)
        # Use Channel 8
        if 'channel_8' not in df.columns:
            raise ValueError("channel_8 column not found in CSV")
            
        raw_data = df['channel_8'].values
        
        # Create Labels based on rule: > 170 is Normal (0), else Damage (1)
        # We label each point first
        point_labels = np.where(raw_data > 170, 0, 1)
        
        # Create Sliding Windows
        # Stride 1
        num_windows = len(raw_data) - window_size + 1
        
        all_samples = []
        all_labels = []
        
        for i in range(num_windows):
            window_data = raw_data[i : i + window_size]
            window_labels = point_labels[i : i + window_size]
            
            # Determine label for the window
            # Strategy: If majority is damage, then damage. Or strict: any damage -> damage.
            # Given user said "depth > 170 is normal", let's assume a strict threshold check on the window.
            # However, for simplicity and robustness, let's use the label of the last point in the window
            # as this is common in time-series forecasting/classification.
            # Or better: majority vote.
            # Let's stick to: Label of the last point.
            label = window_labels[-1]
            
            # Normalize/Preprocess data if needed?
            # HDC usually handles raw values via Levels. 
            # The working synthetic script used raw values (0-170 or 171-250).
            # We keep it as is.
            
            all_samples.append(window_data)
            all_labels.append(label)
            
        # Convert to arrays
        all_samples = np.array(all_samples, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.longlong)
        
        # Shuffle data to ensure mixed classes in train/test
        indices = np.arange(len(all_samples))
        np.random.seed(42) # For reproducibility
        np.random.shuffle(indices)
        all_samples = all_samples[indices]
        all_labels = all_labels[indices]
        
        # Split into Train/Test
        split_idx = int(len(all_samples) * split_ratio)
        
        if is_train:
            self.samples = torch.from_numpy(all_samples[:split_idx]).unsqueeze(2) # [N, Window, 1]
            self.labels = torch.from_numpy(all_labels[:split_idx])
        else:
            self.samples = torch.from_numpy(all_samples[split_idx:]).unsqueeze(2)
            self.labels = torch.from_numpy(all_labels[split_idx:])
            
        print(f"{'Train' if is_train else 'Test'} set size: {len(self.samples)}")
        unique, counts = np.unique(self.labels.numpy(), return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# Encoder
class Channel8Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Channel8Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        # Random hypervectors for channels (we have 1 channel here effectively, or generic channels)
        # In the failing script, it used 1 channel.
        self.channels = embeddings.Random(1, out_features) 
        # Level hypervectors for signal values
        self.signals = embeddings.Level(levels, out_features, high=255.0) # Assuming 0-255 range
        self.n_gram_size = N_GRAM_SIZE

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: [batch, window, 1]
        
        # Signal embedding
        # input is [batch, window, 1]
        # self.signals(input) -> [batch, window, 1, out_features]
        signal = self.signals(input)
        
        # Channel embedding
        # self.channels.weight -> [1, out_features]
        # We need to bind signal (rank 4) with channel (rank ?).
        # We want to broadcast channel over batch and window.
        # channel_hv: [1, 1, 1, out_features]
        channel_hv = self.channels.weight.unsqueeze(0).unsqueeze(0)
        
        # Bind
        samples = torchhd.bind(signal, channel_hv)
        
        # Aggregate (Multiset) over the channel dimension (dim=2)?
        # Actually, here we have only 1 channel.
        # But usually we aggregate over channels to get one HV per time step.
        samples = torchhd.multiset(samples) # Default dim=? 
        # If samples is [batch, window, 1, dim], multiset on dim -2 (channel dim) -> [batch, window, dim]
        # Check torchhd.multiset docs or behavior.
        # If input is [..., n, d], returns [..., d].
        # So [batch, window, 1, dim] -> [batch, window, dim]. Correct.
        
        # N-gram
        sample_hv = torchhd.ngrams(samples, n=self.n_gram_size) # -> [batch, dim] (last ngram?)
        # torchhd.ngrams usually returns [batch, dim] if we process the whole sequence.
        
        return torchhd.hard_quantize(sample_hv)

def main():
    print("Loading Data...")
    csv_path = r"d:\HDC\data\train\接头3.cap_hdc.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    train_ds = RuleBasedDataset(csv_path, window_size=WINDOW, is_train=True)
    test_ds = RuleBasedDataset(csv_path, window_size=WINDOW, is_train=False)
    
    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    encoder = Channel8Encoder(DIMENSIONS, WINDOW, NUM_LEVELS)
    encoder = encoder.to(DEVICE)
    
    model = Centroid(DIMENSIONS, 2) # 0: Normal, 1: Damage
    model = model.to(DEVICE)
    
    # Train
    print("\nTraining...")
    with torch.no_grad():
        for samples, labels in train_loader:
            samples = samples.to(DEVICE)
            labels = labels.to(DEVICE)
            
            samples_hv = encoder(samples)
            model.add(samples_hv, labels)
            
    # Normalize
    if hasattr(model, 'normalize'):
        model.normalize()
        
    # Test
    print("\nTesting...")
    correct = 0
    total = 0
    
    # Class-wise accuracy
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    
    with torch.no_grad():
        for samples, labels in test_loader:
            samples = samples.to(DEVICE)
            labels = labels.to(DEVICE)
            
            samples_hv = encoder(samples)
            outputs = model(samples_hv) # distances or similarities
            predictions = torch.argmax(outputs, dim=1)
            
            # Update totals
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            for p, l in zip(predictions, labels):
                l_item = l.item()
                class_total[l_item] += 1
                if p == l:
                    class_correct[l_item] += 1
            
    print(f"\nOverall Accuracy: {correct / total * 100:.2f}%")
    for cls in [0, 1]:
        if class_total[cls] > 0:
            print(f"Class {cls} ({'Normal' if cls==0 else 'Damage'}) Accuracy: {class_correct[cls] / class_total[cls] * 100:.2f}% ({class_correct[cls]}/{class_total[cls]})")

if __name__ == "__main__":
    main()
