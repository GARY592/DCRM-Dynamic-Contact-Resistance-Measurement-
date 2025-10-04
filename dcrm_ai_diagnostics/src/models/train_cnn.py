"""
1D-CNN model for DCRM pattern recognition.
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


class DCRMDataset(Dataset):
    """Dataset class for DCRM time series data."""
    
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class DCRM1DCNN(nn.Module):
    """1D CNN for DCRM pattern recognition."""
    
    def __init__(self, input_length, num_classes=3):
        super(DCRM1DCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # Assuming input_length is the sequence length
        conv_output_size = self._get_conv_output_size(input_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
    
    def _get_conv_output_size(self, input_length):
        """Calculate the output size after convolutions and pooling."""
        # Simulate forward pass to get output size
        x = torch.zeros(1, 1, input_length)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        # Reshape for 1D convolution: (batch, channels, sequence)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def prepare_cnn_data(features_csv_path: Path, raw_data_dir: Path):
    """
    Prepare data for CNN training by loading raw DCRM signals.
    
    Args:
        features_csv_path: Path to features CSV
        raw_data_dir: Directory containing raw CSV files
    
    Returns:
        Tuple of (X, y) where X is time series data and y is labels
    """
    # Load features to get file names and labels
    features_df = pd.read_csv(features_csv_path)
    
    X = []
    y = []
    
    for _, row in features_df.iterrows():
        file_name = row['file']
        label = row['label']
        
        # Load corresponding raw data
        raw_file = raw_data_dir / file_name
        if raw_file.exists():
            raw_df = pd.read_csv(raw_file)
            
            # Use resistance column as time series
            if 'resistance' in raw_df.columns:
                # Normalize the signal
                signal = raw_df['resistance'].values
                signal = (signal - signal.mean()) / (signal.std() + 1e-8)
                
                # Pad or truncate to fixed length (e.g., 1000 points)
                target_length = 1000
                if len(signal) > target_length:
                    signal = signal[:target_length]
                else:
                    signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
                
                X.append(signal)
                y.append(label)
    
    return np.array(X), np.array(y)


def train_cnn_model():
    """Train the 1D CNN model."""
    # Paths
    PROJECT_ROOT = Path("dcrm_ai_diagnostics")
    FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features_extracted.csv"
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    MODEL_PATH = PROJECT_ROOT / "models" / "cnn_model.pkl"
    SCALER_PATH = PROJECT_ROOT / "models" / "cnn_scaler.pkl"
    
    # Check if features exist
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Features file not found at {FEATURES_CSV}. Run the pipeline first.")
    
    # Prepare data
    print("Preparing CNN data...")
    X, y = prepare_cnn_data(FEATURES_CSV, RAW_DATA_DIR)
    
    if len(X) == 0:
        raise ValueError("No valid raw data found for CNN training.")
    
    print(f"Loaded {len(X)} samples for CNN training")
    
    # Split data
    if len(X) < 2:
        print("Warning: Too few samples for CNN training. Using all data for training.")
        X_train, X_test, y_train, y_test = X, None, y, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
    
    # Create datasets
    train_dataset = DCRMDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    input_length = X_train.shape[1]
    num_classes = len(np.unique(y))
    model = DCRM1DCNN(input_length, num_classes)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Training CNN model...")
    model.train()
    for epoch in range(50):  # Reduced epochs for demo
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluation
    if X_test is not None:
        model.eval()
        with torch.no_grad():
            test_dataset = DCRMDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            all_predictions = []
            all_labels = []
            
            for batch_data, batch_labels in test_loader:
                outputs = model(batch_data)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.numpy())
                all_labels.extend(batch_labels.numpy())
            
            print("CNN Classification Report:")
            print(classification_report(all_labels, all_predictions))
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Save scaler (for consistency with other models)
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"CNN model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    train_cnn_model()
