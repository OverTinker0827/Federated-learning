import React, { useState } from 'react';
import { ServerAPI } from '../services/api';
import './TestPanel.css';

const CATEGORIES = ['electronics', 'fashion', 'home', 'beauty', 'sports'];

const PRESETS = {
  'Wireless earbuds (electronics)': {
    category: 'electronics', price: 79, discount_pct: 15, seller_rating: 4.6, num_reviews: 1200,
    num_photos: 8, shipping_days: 2, free_shipping: 1, views_7d: 650, wishlist_7d: 48,
    promo_active: 1, holiday_season: 0,
  },
  'Linen summer dress (fashion)': {
    category: 'fashion', price: 45, discount_pct: 0, seller_rating: 4.3, num_reviews: 180,
    num_photos: 6, shipping_days: 4, free_shipping: 0, views_7d: 240, wishlist_7d: 15,
    promo_active: 0, holiday_season: 0,
  },
  'Used camping stove (sports, resale)': {
    category: 'sports', price: 28, discount_pct: 0, seller_rating: 3.8, num_reviews: 6,
    num_photos: 2, shipping_days: 8, free_shipping: 0, views_7d: 60, wishlist_7d: 2,
    promo_active: 0, holiday_season: 0,
  },
};

const FIELDS = [
  { key: 'price', label: 'Price (USD)', step: 1 },
  { key: 'discount_pct', label: 'Discount (%)', step: 5 },
  { key: 'seller_rating', label: 'Seller rating (1-5)', step: 0.1 },
  { key: 'num_reviews', label: 'Number of reviews', step: 1 },
  { key: 'num_photos', label: 'Photos on listing', step: 1 },
  { key: 'shipping_days', label: 'Shipping time (days)', step: 1 },
  { key: 'views_7d', label: 'Views (last 7 days)', step: 1 },
  { key: 'wishlist_7d', label: 'Wishlist adds (last 7 days)', step: 1 },
];

const TOGGLES = [
  { key: 'free_shipping', label: 'Free shipping' },
  { key: 'promo_active', label: 'Promoted listing' },
  { key: 'holiday_season', label: 'Holiday season' },
];

function TestPanel() {
  const [listing, setListing] = useState({ ...PRESETS['Wireless earbuds (electronics)'] });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (key, value) => setListing(prev => ({ ...prev, [key]: value }));

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const result = await ServerAPI.predict(listing);
      setPrediction(result);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="test-panel">
      <h2>Model Test Panel</h2>
      <p className="panel-description">
        Describe a marketplace listing to predict how many units it will sell in the next 7 days.
        The DemandNet model was trained with Federated Learning across two laptop sellers and a phone,
        without any seller sharing its raw sales data.
      </p>

      <div className="day-tabs">
        {Object.keys(PRESETS).map(name => (
          <button key={name} className="day-tab" onClick={() => { setListing({ ...PRESETS[name] }); setPrediction(null); }}>
            {name}
          </button>
        ))}
      </div>

      <div className="day-input-form">
        <div className="day-header">
          <h3>Listing details</h3>
        </div>
        <div className="input-grid">
          <div className="input-group">
            <label>Category</label>
            <select value={listing.category} onChange={e => set('category', e.target.value)}>
              {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          {FIELDS.map(f => (
            <div className="input-group" key={f.key}>
              <label>{f.label}</label>
              <input
                type="number"
                step={f.step}
                value={listing[f.key]}
                onChange={e => set(f.key, parseFloat(e.target.value) || 0)}
              />
            </div>
          ))}
          {TOGGLES.map(t => (
            <div className="input-group" key={t.key}>
              <label>{t.label}</label>
              <select value={listing[t.key]} onChange={e => set(t.key, parseInt(e.target.value, 10))}>
                <option value={0}>No</option>
                <option value={1}>Yes</option>
              </select>
            </div>
          ))}
        </div>
      </div>

      <div className="action-buttons">
        <button className="predict-btn" onClick={handlePredict} disabled={loading}>
          {loading ? 'Predicting...' : '🔮 Predict 7-day sales'}
        </button>
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {prediction && (
        <div className="prediction-result">
          <h3>Prediction Result</h3>
          <div className="prediction-value">
            <span className="prediction-number">{prediction.prediction}</span>
            <span className="prediction-unit">units in 7 days</span>
          </div>
          <p className="prediction-message">{prediction.message}</p>
        </div>
      )}
    </div>
  );
}

export default TestPanel;
