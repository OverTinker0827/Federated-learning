#!/usr/bin/env python3
"""
Preprocessing script for Federated Learning Blood Bank Data

This script preprocesses CSV files for all clients, handling:
- Removal of irrelevant features
- Handling NaN values
- Label encoding for categorical columns
- Feature standardization
- Sequence creation for LSTM model
- Saving processed tensors back to client directories

Usage:
    python preprocessing.py                    # Process all clients
    python preprocessing.py --client 1         # Process only client 1
    python preprocessing.py --client 1 2       # Process clients 1 and 2
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional


# Configuration
CLIENT_DIRS = {
    1: "client1",
    2: "client2", 
    3: "client3",
}

# Features to keep for the LSTM model (relevant for blood bank prediction)
# These are the common features across all client datasets for FL consistency
RELEVANT_FEATURES = [
    "DayOfWeek",
    "Month",
    "Weekend",
    "Emergency_Room_Cases",
    "Scheduled_Surgeries",
    "Trauma_Alert_Level",
    "Blood_Type",
    "New_Donations",
    "Units_Used",
    "Starting_Inventory",
    "Ending_Inventory",
    "Days_Supply",
    "Shortage_Flag",
]

# Features to exclude (irrelevant, leaky, or not consistent across clients)
IRRELEVANT_FEATURES = [
    "Date",           # Temporal - only used for sorting
    "Year",           # Can cause data leakage, redundant with Month/DayOfWeek
    "ER_Seasonality", # Derived/engineered feature that may not be in all datasets
    "Expiration_Loss", # May not be present in all datasets
    "Holiday",        # Often constant (0) in some datasets, inconsistent across clients
]

# Target column
TARGET_COLUMN = "Units_Used_tomorrow"

# Categorical columns that need encoding
CATEGORICAL_COLUMNS = ["Blood_Type"]

# Default sequence length for LSTM
DEFAULT_SEQ_LEN = 7


def get_csv_path(client_id: int, base_dir: str = ".") -> str:
    """Get the path to a client's CSV file."""
    client_dir = CLIENT_DIRS.get(client_id)
    if client_dir is None:
        raise ValueError(f"Unknown client ID: {client_id}")
    return os.path.join(base_dir, client_dir, f"blood_bank_data_{client_id}.csv")


def get_output_path(client_id: int, base_dir: str = ".") -> str:
    """Get the output path for processed tensors."""
    client_dir = CLIENT_DIRS.get(client_id)
    if client_dir is None:
        raise ValueError(f"Unknown client ID: {client_id}")
    return os.path.join(base_dir, client_dir, f"blood_bank_data_{client_id}_processed.pt")


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV file and perform initial validation."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def remove_irrelevant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove irrelevant features from the dataframe."""
    cols_to_drop = [col for col in IRRELEVANT_FEATURES if col in df.columns]
    
    if cols_to_drop:
        print(f"  Removing irrelevant features: {cols_to_drop}")
        # Keep Date temporarily for sorting, will drop later
        cols_to_drop_now = [c for c in cols_to_drop if c != "Date"]
        df = df.drop(columns=cols_to_drop_now, errors='ignore')
    
    return df


def handle_missing_values(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Handle missing values in the dataframe."""
    # Report initial NaN counts
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"  Found NaN values:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"    - {col}: {count} NaN(s)")
    
    # Handle target column NaNs first
    if target_col in df.columns:
        target_nans = df[target_col].isna().sum()
        if target_nans > 0:
            print(f"  Handling {target_nans} NaN(s) in target column '{target_col}'")
            # Forward fill, then backward fill for target
            df[target_col] = df[target_col].fillna(method='ffill').fillna(method='bfill')
            # If still NaN, fill with mean
            if df[target_col].isna().any():
                mean_val = df[target_col].mean()
                if pd.isna(mean_val):
                    mean_val = 0.0
                df[target_col] = df[target_col].fillna(mean_val)
    
    # Handle feature NaNs
    feature_cols = [c for c in df.columns if c not in [target_col, "Date"]]
    for col in feature_cols:
        if df[col].isna().any():
            if col in CATEGORICAL_COLUMNS:
                # For categorical, fill with mode
                mode_val = df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
                df[col] = df[col].fillna(fill_val)
            else:
                # For numeric, fill with median (more robust than mean)
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)
    
    # Report remaining NaNs
    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        print(f"  Warning: {remaining_nans} NaN(s) remaining after imputation")
    else:
        print(f"  All NaN values handled successfully")
    
    return df


def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Encode categorical columns using LabelEncoder."""
    encoders = {}
    
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  Encoded '{col}': {len(le.classes_)} unique values")
    
    return df, encoders


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataframe by date if Date column exists."""
    if "Date" in df.columns:
        df = df.sort_values("Date")
        print(f"  Sorted by Date")
        df = df.drop(columns=["Date"])
    return df


def remove_constant_columns(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Remove columns with zero variance (constant values)."""
    feature_cols = [c for c in df.columns if c != target_col]
    
    constant_cols = []
    for col in feature_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"  Removing constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    
    return df


def convert_to_numeric(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Convert all feature columns to numeric, coercing errors to NaN."""
    feature_cols = [c for c in df.columns if c != target_col]
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Also ensure target is numeric
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    return df


def standardize_features(features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, scaler


def create_sequences(data: np.ndarray, target: np.ndarray, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create sequences for LSTM model."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(target[i + seq_len])
    
    # Convert to numpy arrays first for better performance
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.float32)
    
    return torch.from_numpy(X_array), torch.from_numpy(y_array)


def validate_tensors(X: torch.Tensor, y: torch.Tensor) -> bool:
    """Validate that tensors don't contain NaN or Inf values."""
    has_nan = torch.isnan(X).any() or torch.isnan(y).any()
    has_inf = torch.isinf(X).any() or torch.isinf(y).any()
    
    if has_nan:
        print(f"  Warning: Tensors contain NaN values!")
    if has_inf:
        print(f"  Warning: Tensors contain Inf values!")
    
    return not (has_nan or has_inf)


def preprocess_client(client_id: int, base_dir: str = ".", seq_len: int = DEFAULT_SEQ_LEN) -> dict:
    """
    Preprocess data for a single client.
    
    Returns dict with processing statistics and output path.
    """
    print(f"\n{'='*60}")
    print(f"Processing Client {client_id}")
    print(f"{'='*60}")
    
    result = {
        "client_id": client_id,
        "success": False,
        "error": None,
        "output_path": None,
        "stats": {}
    }
    
    try:
        # Load CSV
        csv_path = get_csv_path(client_id, base_dir)
        print(f"\n[1/8] Loading CSV: {csv_path}")
        df = load_csv(csv_path)
        result["stats"]["original_rows"] = len(df)
        result["stats"]["original_cols"] = len(df.columns)
        
        # Validate target column exists
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV")
        
        # Sort by date first (before any processing)
        print(f"\n[2/8] Sorting by date")
        df = sort_by_date(df)
        
        # Remove irrelevant features
        print(f"\n[3/8] Removing irrelevant features")
        df = remove_irrelevant_features(df)
        
        # Encode categorical columns
        print(f"\n[4/8] Encoding categorical columns")
        df, encoders = encode_categorical(df)
        
        # Convert to numeric
        print(f"\n[5/8] Converting to numeric")
        df = convert_to_numeric(df)
        
        # Handle missing values
        print(f"\n[6/8] Handling missing values")
        df = handle_missing_values(df)
        
        # Remove constant columns
        print(f"\n[7/8] Removing constant/zero-variance columns")
        df = remove_constant_columns(df)
        
        # Separate features and target
        target = df[TARGET_COLUMN].values.astype(float)
        features_df = df.drop(columns=[TARGET_COLUMN])
        
        # Handle any remaining NaNs in features
        features_df = features_df.fillna(features_df.mean())
        features = features_df.values
        
        # Standardize features
        print(f"\n[8/8] Standardizing features and creating sequences")
        features_scaled, scaler = standardize_features(features)
        
        # Create sequences
        X_seq, y_seq = create_sequences(features_scaled, target, seq_len)
        
        # Validate tensors
        if not validate_tensors(X_seq, y_seq):
            # Try to fix by replacing NaN/Inf
            X_seq = torch.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
            y_seq = torch.nan_to_num(y_seq, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"  Applied nan_to_num correction")
        
        # Save processed tensors
        output_path = get_output_path(client_id, base_dir)
        torch.save({
            "X_train": X_seq,
            "y_train": y_seq,
            "feature_names": list(features_df.columns),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "seq_len": seq_len,
        }, output_path)
        
        # Update result
        result["success"] = True
        result["output_path"] = output_path
        result["stats"]["processed_rows"] = len(df)
        result["stats"]["num_features"] = features.shape[1]
        result["stats"]["num_sequences"] = len(X_seq)
        result["stats"]["sequence_shape"] = list(X_seq.shape)
        
        print(f"\n✓ Successfully processed Client {client_id}")
        print(f"  Output: {output_path}")
        print(f"  Sequences: {X_seq.shape[0]}")
        print(f"  Sequence shape: (batch, seq_len={seq_len}, features={X_seq.shape[2]})")
        print(f"  Feature columns: {list(features_df.columns)}")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"\n✗ Error processing Client {client_id}: {e}")
    
    return result


def preprocess_all_clients(base_dir: str = ".", seq_len: int = DEFAULT_SEQ_LEN, 
                          client_ids: Optional[List[int]] = None) -> List[dict]:
    """
    Preprocess data for all or specified clients.
    
    Args:
        base_dir: Base directory containing client folders
        seq_len: Sequence length for LSTM
        client_ids: List of client IDs to process, or None for all
    
    Returns:
        List of result dictionaries for each client
    """
    if client_ids is None:
        client_ids = list(CLIENT_DIRS.keys())
    
    print("=" * 60)
    print("Federated Learning Data Preprocessing")
    print("=" * 60)
    print(f"Base directory: {os.path.abspath(base_dir)}")
    print(f"Sequence length: {seq_len}")
    print(f"Clients to process: {client_ids}")
    print(f"Target column: {TARGET_COLUMN}")
    
    results = []
    for client_id in client_ids:
        result = preprocess_client(client_id, base_dir, seq_len)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"  ✓ Client {r['client_id']}: {r['stats']['num_sequences']} sequences, "
              f"{r['stats']['num_features']} features")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  ✗ Client {r['client_id']}: {r['error']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess blood bank CSV data for federated learning clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python preprocessing.py                    # Process all clients
    python preprocessing.py --client 1         # Process only client 1
    python preprocessing.py --client 1 2       # Process clients 1 and 2
    python preprocessing.py --seq-len 14       # Use 14-day sequences
        """
    )
    parser.add_argument(
        "--client", 
        type=int, 
        nargs="+",
        help="Client ID(s) to process. If not specified, processes all clients."
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Sequence length for LSTM (default: {DEFAULT_SEQ_LEN})"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory containing client folders (default: current directory)"
    )
    
    args = parser.parse_args()
    
    results = preprocess_all_clients(
        base_dir=args.base_dir,
        seq_len=args.seq_len,
        client_ids=args.client
    )
    
    # Exit with error code if any client failed
    if any(not r["success"] for r in results):
        exit(1)


if __name__ == "__main__":
    main()
