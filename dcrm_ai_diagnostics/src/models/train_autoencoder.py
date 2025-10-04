"""
Autoencoder model for DCRM anomaly detection.
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
from sklearn.metrics import roc_auc_score
import joblib


class DCRMAutoencoderDataset(Dataset):
    """Dataset class for DCRM Autoencoder data."""
    
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Autoencoder: input = target


class DCRMAutoencoder(nn.Module):
    """Autoencoder for DCRM anomaly detection."""
    
    def __init__(self, input_size, encoding_dim=32):
        super(DCRMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_size),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get encoded representation."""
        return self.encoder(x)
    
    def decode(self, encoded):
        """Decode from encoded representation."""
        return self.decoder(encoded)


def prepare_autoencoder_data(features_csv_path: Path):
    """
    Prepare data for Autoencoder training using extracted features.
    
    Args:
        features_csv_path: Path to features CSV
    
    Returns:
        Feature matrix X
    """
    # Load features
    features_df = pd.read_csv(features_csv_path)
    
    # Drop non-feature columns
    feature_columns = [col for col in features_df.columns if col not in ['file', 'label']]
    X = features_df[feature_columns].values
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, scaler


def train_autoencoder_model():
    """Train the Autoencoder model."""
    # Paths
    PROJECT_ROOT = Path("dcrm_ai_diagnostics")
    FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features_extracted.csv"
    MODEL_PATH = PROJECT_ROOT / "models" / "autoencoder_model.pkl"
    SCALER_PATH = PROJECT_ROOT / "models" / "autoencoder_scaler.pkl"
    
    # Check if features exist
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Features file not found at {FEATURES_CSV}. Run the pipeline first.")
    
    # Prepare data
    print("Preparing Autoencoder data...")
    X, scaler = prepare_autoencoder_data(FEATURES_CSV)
    
    if len(X) == 0:
        raise ValueError("No valid features found for Autoencoder training.")
    
    print(f"Loaded {len(X)} samples for Autoencoder training")
    
    # Split data (use all for training since it's unsupervised)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = DCRMAutoencoderDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    input_size = X.shape[1]
    encoding_dim = min(32, input_size // 2)  # Encoding dimension
    model = DCRMAutoencoder(input_size, encoding_dim)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    print("Training Autoencoder model...")
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(200):  # More epochs for autoencoder
        total_loss = 0
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_target)
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
        
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Evaluation on test set
    if X_test is not None:
        model.eval()
        with torch.no_grad():
            test_dataset = DCRMAutoencoderDataset(X_test)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            total_test_loss = 0
            for batch_data, batch_target in test_loader:
                reconstructed = model(batch_data)
                loss = criterion(reconstructed, batch_target)
                total_test_loss += loss.item()
            
            avg_test_loss = total_test_loss / len(test_loader)
            print(f"Test Reconstruction Loss: {avg_test_loss:.6f}")
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Autoencoder model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


def detect_anomalies_with_autoencoder(model, scaler, X, threshold_percentile=95):
    """
    Detect anomalies using trained autoencoder.
    
    Args:
        model: Trained autoencoder model
        scaler: Fitted scaler
        X: Input data
        threshold_percentile: Percentile for anomaly threshold
    
    Returns:
        Tuple of (anomaly_scores, anomaly_predictions)
    """
    model.eval()
    with torch.no_grad():
        # Normalize data
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Get reconstructions
        reconstructed = model(X_tensor)
        
        # Calculate reconstruction error
        reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        anomaly_scores = reconstruction_error.numpy()
        
        # Determine threshold
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        
        # Predict anomalies
        anomaly_predictions = (anomaly_scores > threshold).astype(int)
    
    return anomaly_scores, anomaly_predictions


if __name__ == "__main__":
    train_autoencoder_model()
