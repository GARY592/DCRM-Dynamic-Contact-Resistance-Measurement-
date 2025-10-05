"""
LSTM model for DCRM time-series analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


class DCRMLSTMDataset(Dataset):
    """Dataset class for DCRM LSTM data."""
    
    def __init__(self, data, labels, sequence_length=50):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # self.data is expected to be shaped (num_sequences, sequence_length) after prepare step
        # When prepared, we will pass sequences directly instead of slicing here
        sequence = self.data[idx]
        label = self.labels[idx]
        # Return shape (sequence_length, 1) for single feature
        sequence = torch.FloatTensor(sequence).view(-1, 1)
        return sequence, torch.LongTensor([label])


class DCRMLSTM(nn.Module):
    """LSTM model for DCRM time-series analysis."""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(DCRMLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(attended_output)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def prepare_lstm_data(features_csv_path: Path, raw_data_dir: Path, sequence_length=50):
    """
    Prepare data for LSTM training by creating sequences from raw DCRM signals.
    
    Args:
        features_csv_path: Path to features CSV
        raw_data_dir: Directory containing raw CSV files
        sequence_length: Length of sequences for LSTM
    
    Returns:
        Tuple of (X, y) where X is sequences and y is labels
    """
    # Load features to get file names and labels
    features_df = pd.read_csv(features_csv_path)
    
    X = []
    y = []
    
    for _, row in features_df.iterrows():
        file_name = row['file'] if 'file' in row else None
        # Derive label if missing: healthy_* -> 0, faulty_* -> 1, else 0
        if 'label' in row.index:
            label = row['label']
        else:
            if isinstance(file_name, str):
                if file_name.lower().startswith('healthy_'):
                    label = 0
                elif file_name.lower().startswith('faulty_'):
                    label = 1
                else:
                    label = 0
            else:
                label = 0
        
        # Load corresponding raw data
        raw_file = raw_data_dir / file_name
        if raw_file.exists():
            raw_df = pd.read_csv(raw_file)
            
            # Use resistance column as time series
            if 'resistance' in raw_df.columns:
                signal = raw_df['resistance'].values
                signal = (signal - signal.mean()) / (signal.std() + 1e-8)
                # Always create non-overlapping sequences to avoid massive 4D shapes
                if len(signal) < sequence_length:
                    seq = np.pad(signal, (0, sequence_length - len(signal)), mode='constant')
                    X.append(seq)
                    y.append(label)
                else:
                    for i in range(0, len(signal) - sequence_length + 1, sequence_length):
                        seq = signal[i:i + sequence_length]
                        if len(seq) == sequence_length:
                            X.append(seq)
                            y.append(label)
    
    return np.array(X), np.array(y)


def train_lstm_model():
    """Train the LSTM model."""
    # Paths
    PROJECT_ROOT = Path("dcrm_ai_diagnostics")
    FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features_extracted.csv"
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.pkl"
    SCALER_PATH = PROJECT_ROOT / "models" / "lstm_scaler.pkl"
    
    # Check if features exist
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Features file not found at {FEATURES_CSV}. Run the pipeline first.")
    
    # Prepare data
    print("Preparing LSTM data...")
    sequence_length = 50
    X, y = prepare_lstm_data(FEATURES_CSV, RAW_DATA_DIR, sequence_length)
    
    if len(X) == 0:
        raise ValueError("No valid raw data found for LSTM training.")
    
    print(f"Loaded {len(X)} sequences for LSTM training")
    
    # Reshape for LSTM: (samples, sequence_length) -> dataset returns (seq_len, 1)
    X = X.reshape(X.shape[0], sequence_length)
    
    # Split data
    if len(X) < 2:
        print("Warning: Too few samples for LSTM training. Using all data for training.")
        X_train, X_test, y_train, y_test = X, None, y, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
    
    # Create datasets
    train_dataset = DCRMLSTMDataset(X_train, y_train, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    input_size = 1  # Single feature (resistance)
    hidden_size = 64
    num_layers = 2
    num_classes = len(np.unique(y))
    model = DCRMLSTM(input_size, hidden_size, num_layers, num_classes)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    print("Training LSTM model...")
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(100):  # More epochs for LSTM
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    if X_test is not None:
        model.eval()
        with torch.no_grad():
            test_dataset = DCRMLSTMDataset(X_test, y_test, sequence_length)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            all_predictions = []
            all_labels = []
            
            for batch_data, batch_labels in test_loader:
                outputs = model(batch_data)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.numpy())
                all_labels.extend(batch_labels.squeeze().numpy())
            
            print("LSTM Classification Report:")
            print(classification_report(all_labels, all_predictions))
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Save scaler
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(X_train.shape[0], -1))
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"LSTM model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    train_lstm_model()
