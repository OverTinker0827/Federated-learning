#!/usr/bin/env python3
"""
Generate mock e-commerce marketplace data for the federated learning demo.

Each participant is a different seller with its own (non-IID) catalogue:

    client1  TechHub Electronics         mostly electronics, premium prices
    client2  Urban Threads Boutique      fashion + beauty
    client3  Casa & Garden Outlet        home + sports (optional 4th client)
    phone    Grama Connect (on-device)   village marketplace seller on a phone
    server   held-out marketplace sample used to evaluate the global model

All sellers share the same underlying "market" demand behaviour, so a
federated model trained on everyone generalises better than any single
seller's local model.

Outputs
    clientN/ecommerce_data_N.csv              raw listings
    clientN/ecommerce_data_N_processed.pt     {X_train, y_train, metadata}
    phone_client/ecommerce_phone_data.csv     raw listings kept "on the phone"
    phone_client/phone_data.json              normalised arrays for the JS trainer
    server/test_data.pt                       {X_test, y_test, metadata}
    ecommerce_scaler.json                     shared normalisation constants

Usage:
    python generate_ecommerce_data.py [--seed 7]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import ecommerce_features as ef

ROOT = os.path.dirname(os.path.abspath(__file__))

CATEGORY_REF_PRICE = {"electronics": 220.0, "fashion": 45.0, "home": 60.0, "beauty": 22.0, "sports": 70.0}
CATEGORY_BIAS = {"electronics": 0.10, "fashion": 0.25, "home": 0.0, "beauty": 0.30, "sports": -0.05}
HOLIDAY_SENSITIVITY = {"electronics": 0.55, "fashion": 0.40, "home": 0.25, "beauty": 0.35, "sports": 0.20}

SEGMENTS = {
    "client1": dict(
        name="TechHub Electronics", n=1600,
        cat_p={"electronics": 0.72, "home": 0.12, "sports": 0.08, "fashion": 0.04, "beauty": 0.04},
        price_mult=1.25, rating=(4.4, 0.25), reviews_scale=600, photos=(6, 11), ship=(1, 5),
        free_ship_p=0.7, promo_p=0.25,
    ),
    "client2": dict(
        name="Urban Threads Boutique", n=1500,
        cat_p={"fashion": 0.58, "beauty": 0.32, "home": 0.05, "sports": 0.03, "electronics": 0.02},
        price_mult=1.0, rating=(4.2, 0.35), reviews_scale=250, photos=(3, 9), ship=(2, 7),
        free_ship_p=0.5, promo_p=0.3,
    ),
    "client3": dict(
        name="Casa & Garden Outlet", n=1400,
        cat_p={"home": 0.6, "sports": 0.28, "electronics": 0.06, "fashion": 0.03, "beauty": 0.03},
        price_mult=0.9, rating=(4.0, 0.4), reviews_scale=150, photos=(2, 8), ship=(3, 10),
        free_ship_p=0.35, promo_p=0.15,
    ),
    "phone": dict(
        name="Grama Connect", n=1200,
        cat_p={"home": 0.28, "sports": 0.27, "electronics": 0.22, "fashion": 0.18, "beauty": 0.05},
        price_mult=0.55, rating=(3.9, 0.5), reviews_scale=40, photos=(1, 5), ship=(3, 12),
        free_ship_p=0.2, promo_p=0.08,
    ),
    "server": dict(
        name="Marketplace holdout", n=1500,
        cat_p={c: 0.2 for c in ef.CATEGORIES},
        price_mult=1.0, rating=(4.15, 0.4), reviews_scale=250, photos=(1, 11), ship=(1, 12),
        free_ship_p=0.45, promo_p=0.18,
    ),
}


def make_listings(seg_key, seg, rng):
    n = seg["n"]
    cats = rng.choice(list(seg["cat_p"].keys()), size=n, p=np.array(list(seg["cat_p"].values())))
    ref = np.array([CATEGORY_REF_PRICE[c] for c in cats])
    price = np.round(ref * seg["price_mult"] * rng.lognormal(0.0, 0.45, n), 2)
    price = np.clip(price, 2.0, 3000.0)
    discount = np.where(rng.random(n) < 0.45, rng.choice([5, 10, 15, 20, 25, 30, 40, 50], size=n), 0)
    rating = np.clip(rng.normal(*seg["rating"], n), 1.0, 5.0).round(1)
    reviews = rng.poisson(seg["reviews_scale"] * rng.lognormal(0, 1.0, n)).astype(int)
    photos = rng.integers(seg["photos"][0], seg["photos"][1] + 1, n)
    ship = rng.integers(seg["ship"][0], seg["ship"][1] + 1, n)
    free = (rng.random(n) < seg["free_ship_p"]).astype(int)
    promo = (rng.random(n) < seg["promo_p"]).astype(int)
    holiday = (rng.random(n) < 0.22).astype(int)

    # Traffic depends on visibility factors
    views_lambda = np.exp(
        3.6 + 0.9 * promo + 0.06 * photos + 0.25 * (rating - 4) + 0.12 * np.log1p(reviews)
        + 0.35 * holiday + rng.normal(0, 0.45, n)
    )
    views = rng.poisson(views_lambda)
    wish_rate = np.clip(0.04 + 0.002 * discount + 0.02 * (rating - 4) + rng.normal(0, 0.015, n), 0.005, 0.3)
    wishlist = rng.binomial(views, wish_rate)

    # Ground-truth demand model (shared by every seller)
    eff_disc = 0.035 * np.minimum(discount, 30) + 0.008 * np.maximum(discount - 30, 0)
    log_lam = (
        -2.2
        + 0.55 * np.log1p(views)
        + 0.30 * np.log1p(wishlist)
        - 0.85 * np.log(price / ref)
        + eff_disc
        + 0.55 * (rating - 4.0)
        + 0.10 * np.log1p(reviews)
        + 0.04 * photos
        - 0.07 * ship
        + 0.25 * free
        + 0.20 * promo
        + holiday * np.array([HOLIDAY_SENSITIVITY[c] for c in cats])
        + np.array([CATEGORY_BIAS[c] for c in cats])
        + rng.normal(0, 0.18, n)
    )
    units = rng.poisson(np.exp(np.minimum(log_lam, 4.8)))

    prefix = {"client1": "TH", "client2": "UT", "client3": "CG", "phone": "PR", "server": "MK"}[seg_key]
    return pd.DataFrame({
        "listing_id": [f"{prefix}-{i:05d}" for i in range(n)],
        "category": cats,
        "price": price,
        "discount_pct": discount,
        "seller_rating": rating,
        "num_reviews": reviews,
        "num_photos": photos,
        "shipping_days": ship,
        "free_shipping": free,
        "views_7d": views,
        "wishlist_7d": wishlist,
        "promo_active": promo,
        "holiday_season": holiday,
        ef.TARGET: units,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    frames = {k: make_listings(k, seg, rng) for k, seg in SEGMENTS.items()}

    # Pre-agreed normalisation (in a real deployment these would be published
    # constants; here we derive them once from a pooled sample)
    pooled = pd.concat(frames.values(), ignore_index=True)
    scaler = ef.fit_scaler(ef.df_to_numeric(pooled), pooled[ef.TARGET].values.astype(np.float32))
    ef.save_scaler(scaler)
    print(f"Saved scaler -> {ef.SCALER_PATH}")

    def tensors(df):
        X = torch.tensor(ef.transform_x(ef.df_to_numeric(df), scaler))
        y = torch.tensor(ef.transform_y(df[ef.TARGET].values, scaler))
        return X, y

    meta_common = {"feature_names": ef.FEATURE_NAMES, "feature_dim": ef.NUM_FEATURES,
                   "y_mean": scaler["y_mean"], "y_std": scaler["y_std"], "task": "units_sold_next_7d"}

    for idx in (1, 2, 3):
        key = f"client{idx}"
        df = frames[key]
        d = os.path.join(ROOT, key)
        df.to_csv(os.path.join(d, f"ecommerce_data_{idx}.csv"), index=False)
        X, y = tensors(df)
        torch.save({"X_train": X, "y_train": y, "metadata": {**meta_common, "store": SEGMENTS[key]["name"]}},
                   os.path.join(d, f"ecommerce_data_{idx}_processed.pt"))
        print(f"{key:8s} {SEGMENTS[key]['name']:24s} {len(df):5d} listings, mean units {df[ef.TARGET].mean():.1f}")

    # Phone: raw CSV + JSON consumed by the browser trainer
    df = frames["phone"]
    pdir = os.path.join(ROOT, "phone_client")
    os.makedirs(pdir, exist_ok=True)
    df.to_csv(os.path.join(pdir, "ecommerce_phone_data.csv"), index=False)
    X, y = tensors(df)
    phone_json = {
        "store": SEGMENTS["phone"]["name"],
        "feature_names": ef.FEATURE_NAMES,
        "y_mean": scaler["y_mean"],
        "y_std": scaler["y_std"],
        "X": np.round(X.numpy(), 5).tolist(),
        "y": np.round(y.numpy(), 5).tolist(),
        "raw": df.to_dict("records"),
    }
    with open(os.path.join(pdir, "phone_data.json"), "w") as fh:
        json.dump(phone_json, fh, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"phone    {SEGMENTS['phone']['name']:24s} {len(df):5d} listings, mean units {df[ef.TARGET].mean():.1f}")

    df = frames["server"]
    X, y = tensors(df)
    torch.save({"X_test": X, "y_test": y, "metadata": meta_common}, os.path.join(ROOT, "server", "test_data.pt"))
    df.to_csv(os.path.join(ROOT, "server", "ecommerce_test_data.csv"), index=False)
    print(f"server   test set                 {len(df):5d} listings")


if __name__ == "__main__":
    main()
