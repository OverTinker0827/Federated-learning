"""
Shared feature definition for the e-commerce demand-forecasting task.

Task: given a marketplace listing's current state, predict how many units
it will sell in the next 7 days (regression).

Every participant (laptop clients, phone client, server) uses exactly the
same feature order and the same pre-agreed normalisation constants stored in
ecommerce_scaler.json, so that federated averaging operates on a consistent
input space.
"""

import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(ROOT, "ecommerce_scaler.json")

CATEGORIES = ["electronics", "fashion", "home", "beauty", "sports"]

# Raw listing fields (what a seller app would store locally)
RAW_FIELDS = [
    "listing_id",
    "category",
    "price",
    "discount_pct",
    "seller_rating",
    "num_reviews",
    "num_photos",
    "shipping_days",
    "free_shipping",
    "views_7d",
    "wishlist_7d",
    "promo_active",
    "holiday_season",
]

TARGET = "units_sold_next_7d"

# Model input features, in order
NUMERIC_FEATURES = [
    "log_price",
    "discount_pct",
    "seller_rating",
    "log_num_reviews",
    "num_photos",
    "shipping_days",
    "free_shipping",
    "log_views_7d",
    "log_wishlist_7d",
    "promo_active",
    "holiday_season",
]
FEATURE_NAMES = NUMERIC_FEATURES + [f"cat_{c}" for c in CATEGORIES]
NUM_FEATURES = len(FEATURE_NAMES)

FIELD_DESCRIPTIONS = {
    "category": "Product category",
    "price": "Listing price (USD)",
    "discount_pct": "Current discount (%)",
    "seller_rating": "Seller rating (1-5 stars)",
    "num_reviews": "Number of product reviews",
    "num_photos": "Photos on the listing",
    "shipping_days": "Estimated delivery time (days)",
    "free_shipping": "Free shipping (0/1)",
    "views_7d": "Listing views in the last 7 days",
    "wishlist_7d": "Wishlist adds in the last 7 days",
    "promo_active": "Featured / promoted listing (0/1)",
    "holiday_season": "Holiday shopping season (0/1)",
}


def raw_to_numeric(row):
    """Turn one raw listing (dict-like) into the unscaled feature vector."""
    cat = str(row["category"]).lower()
    vec = [
        math.log1p(float(row["price"])),
        float(row["discount_pct"]),
        float(row["seller_rating"]),
        math.log1p(float(row["num_reviews"])),
        float(row["num_photos"]),
        float(row["shipping_days"]),
        float(row["free_shipping"]),
        math.log1p(float(row["views_7d"])),
        math.log1p(float(row["wishlist_7d"])),
        float(row["promo_active"]),
        float(row["holiday_season"]),
    ]
    vec += [1.0 if cat == c else 0.0 for c in CATEGORIES]
    return vec


def df_to_numeric(df):
    return np.array([raw_to_numeric(r) for r in df.to_dict("records")], dtype=np.float32)


def fit_scaler(X, y):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # one-hot columns: keep as 0/1 (no scaling) so they stay interpretable
    n_num = len(NUMERIC_FEATURES)
    mean[n_num:] = 0.0
    std[n_num:] = 1.0
    std[std < 1e-6] = 1.0
    return {
        "feature_names": FEATURE_NAMES,
        "x_mean": mean.tolist(),
        "x_std": std.tolist(),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }


def save_scaler(scaler, path=SCALER_PATH):
    with open(path, "w") as fh:
        json.dump(scaler, fh, indent=2)


def load_scaler(path=SCALER_PATH):
    with open(path) as fh:
        return json.load(fh)


def transform_x(X, scaler):
    return ((np.asarray(X, dtype=np.float32) - np.array(scaler["x_mean"], dtype=np.float32))
            / np.array(scaler["x_std"], dtype=np.float32)).astype(np.float32)


def transform_y(y, scaler):
    return ((np.asarray(y, dtype=np.float32) - scaler["y_mean"]) / scaler["y_std"]).astype(np.float32)


def inverse_y(y_scaled, scaler):
    return np.asarray(y_scaled) * scaler["y_std"] + scaler["y_mean"]
