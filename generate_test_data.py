#!/usr/bin/env python3
"""
Generate Test Data for Federated Learning Server

This script samples data from each client's preprocessed dataset and combines
them into a single test dataset for model evaluation after training.

The test data is saved in the server directory as test_data.pt

Usage:
    python generate_test_data.py                    # Default: 100 samples per client
    python generate_test_data.py --samples 50      # 50 samples per client
    python generate_test_data.py --seed 42         # Set random seed
"""

import argparse
import os
import torch
import numpy as np
from typing import Dict, List, Tuple

# Configuration
CLIENT_DIRS = {
    1: "client1",
    2: "client2",
    3: "client3",
}

SERVER_DIR = "server"
OUTPUT_FILE = "test_data.pt"

DEFAULT_SAMPLES_PER_CLIENT = 100
DEFAULT_SEED = 42


def load_processed_data(client_id: int, base_dir: str = ".") -> Tuple[torch.Tensor, torch.Tensor]:
    """Load preprocessed data for a client."""
    client_dir = CLIENT_DIRS.get(client_id)
    if client_dir is None:
        raise ValueError(f"Unknown client ID: {client_id}")
    
    pt_path = os.path.join(base_dir, client_dir, f"blood_bank_data_{client_id}_processed.pt")
    
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Processed data not found: {pt_path}")
    
    data = torch.load(pt_path, weights_only=False)
    return data["X_train"], data["y_train"]


def sample_data(X: torch.Tensor, y: torch.Tensor, num_samples: int, 
                seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample data from a dataset."""
    total_samples = len(X)
    
    if num_samples >= total_samples:
        print(f"  Warning: Requested {num_samples} samples but only {total_samples} available")
        num_samples = total_samples
    
    if seed is not None:
        np.random.seed(seed)
    
    indices = np.random.choice(total_samples, size=num_samples, replace=False)
    indices = np.sort(indices)  # Keep temporal order
    
    return X[indices], y[indices]


def generate_test_data(base_dir: str = ".", 
                       samples_per_client: int = DEFAULT_SAMPLES_PER_CLIENT,
                       seed: int = DEFAULT_SEED,
                       client_ids: List[int] = None) -> Dict:
    """
    Generate test data by sampling from each client's preprocessed data.
    
    Args:
        base_dir: Base directory containing client folders
        samples_per_client: Number of samples to take from each client
        seed: Random seed for reproducibility
        client_ids: List of client IDs to sample from, or None for all
    
    Returns:
        Dictionary containing combined test data and metadata
    """
    if client_ids is None:
        client_ids = list(CLIENT_DIRS.keys())
    
    print("=" * 60)
    print("Generating Test Data for FL Server")
    print("=" * 60)
    print(f"Base directory: {os.path.abspath(base_dir)}")
    print(f"Samples per client: {samples_per_client}")
    print(f"Random seed: {seed}")
    print(f"Clients: {client_ids}")
    
    all_X = []
    all_y = []
    client_sample_info = {}
    feature_dim = None
    seq_len = None
    
    for client_id in client_ids:
        print(f"\nProcessing Client {client_id}...")
        
        try:
            X, y = load_processed_data(client_id, base_dir)
            print(f"  Loaded data: X shape {X.shape}, y shape {y.shape}")
            
            # Verify feature dimensions are consistent
            if feature_dim is None:
                feature_dim = X.shape[2]
                seq_len = X.shape[1]
            elif X.shape[2] != feature_dim:
                print(f"  Warning: Client {client_id} has {X.shape[2]} features, expected {feature_dim}")
                print(f"  Skipping this client to maintain consistency")
                continue
            
            # Sample data
            X_sampled, y_sampled = sample_data(X, y, samples_per_client, seed + client_id)
            print(f"  Sampled: X shape {X_sampled.shape}, y shape {y_sampled.shape}")
            
            all_X.append(X_sampled)
            all_y.append(y_sampled)
            
            client_sample_info[client_id] = {
                "total_samples": len(X),
                "sampled_samples": len(X_sampled),
                "feature_dim": int(X.shape[2]),
                "seq_len": int(X.shape[1])
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not all_X:
        raise ValueError("No data could be loaded from any client")
    
    # Combine all sampled data
    X_test = torch.cat(all_X, dim=0)
    y_test = torch.cat(all_y, dim=0)
    
    print(f"\nCombined test data:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    # Validate data
    if torch.isnan(X_test).any() or torch.isinf(X_test).any():
        print("  Warning: X_test contains NaN or Inf values")
        X_test = torch.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    if torch.isnan(y_test).any() or torch.isinf(y_test).any():
        print("  Warning: y_test contains NaN or Inf values")
        y_test = torch.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create result dictionary
    result = {
        "X_test": X_test,
        "y_test": y_test,
        "metadata": {
            "total_samples": int(len(X_test)),
            "feature_dim": int(feature_dim),
            "seq_len": int(seq_len),
            "samples_per_client": samples_per_client,
            "seed": seed,
            "clients": client_ids,
            "client_info": client_sample_info
        }
    }
    
    return result


def save_test_data(data: Dict, base_dir: str = ".") -> str:
    """Save test data to the server directory."""
    output_path = os.path.join(base_dir, SERVER_DIR, OUTPUT_FILE)
    
    # Ensure server directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(data, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for federated learning server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_test_data.py                    # Default: 100 samples per client
    python generate_test_data.py --samples 50      # 50 samples per client
    python generate_test_data.py --client 1 2      # Sample from clients 1 and 2 only
    python generate_test_data.py --seed 123        # Use different random seed
        """
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLIENT,
        help=f"Number of samples per client (default: {DEFAULT_SAMPLES_PER_CLIENT})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--client",
        type=int,
        nargs="+",
        help="Client ID(s) to sample from. If not specified, samples from all clients."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory containing client and server folders (default: current directory)"
    )
    
    args = parser.parse_args()
    
    try:
        # Generate test data
        data = generate_test_data(
            base_dir=args.base_dir,
            samples_per_client=args.samples,
            seed=args.seed,
            client_ids=args.client
        )
        
        # Save to server directory
        output_path = save_test_data(data, args.base_dir)
        
        print(f"\n{'=' * 60}")
        print("TEST DATA GENERATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Output saved to: {output_path}")
        print(f"Total test samples: {data['metadata']['total_samples']}")
        print(f"Feature dimension: {data['metadata']['feature_dim']}")
        print(f"Sequence length: {data['metadata']['seq_len']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)


if __name__ == "__main__":
    main()
