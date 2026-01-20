import React, { useState, useEffect } from 'react';
import { ServerAPI } from '../services/api';
import './TestPanel.css';

const BLOOD_TYPES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-'];
const BLOOD_TYPE_ENCODING = {
  'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3,
  'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7
};

const DEFAULT_DAY = {
  DayOfWeek: 0,
  Month: 1,
  Weekend: 0,
  Emergency_Room_Cases: 50,
  Scheduled_Surgeries: 10,
  Trauma_Alert_Level: 1,
  Blood_Type: 'O+',
  New_Donations: 20,
  Units_Used: 30,
  Starting_Inventory: 100,
  Ending_Inventory: 90,
  Days_Supply: 5,
  Shortage_Flag: 0
};

function TestPanel() {
  const [days, setDays] = useState(Array(7).fill(null).map(() => ({ ...DEFAULT_DAY })));
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeDay, setActiveDay] = useState(0);

  const handleInputChange = (dayIndex, field, value) => {
    const newDays = [...days];
    let parsedValue = value;

    // Handle numeric fields
    if (field !== 'Blood_Type') {
      parsedValue = parseFloat(value) || 0;
    }

    newDays[dayIndex] = {
      ...newDays[dayIndex],
      [field]: parsedValue
    };
    setDays(newDays);
  };

  const prepareFeatures = () => {
    // Convert days to feature array format expected by the model
    // Order must match: DayOfWeek, Month, Weekend, Emergency_Room_Cases, Scheduled_Surgeries,
    // Trauma_Alert_Level, Blood_Type, New_Donations, Units_Used, Starting_Inventory,
    // Ending_Inventory, Days_Supply, Shortage_Flag
    return days.map(day => [
      day.DayOfWeek,
      day.Month,
      day.Weekend,
      day.Emergency_Room_Cases,
      day.Scheduled_Surgeries,
      day.Trauma_Alert_Level,
      BLOOD_TYPE_ENCODING[day.Blood_Type] || 0,
      day.New_Donations,
      day.Units_Used,
      day.Starting_Inventory,
      day.Ending_Inventory,
      day.Days_Supply,
      day.Shortage_Flag
    ]);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const features = prepareFeatures();
      const result = await ServerAPI.predict(features);
      setPrediction(result);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  const copyFromPreviousDay = (dayIndex) => {
    if (dayIndex > 0) {
      const newDays = [...days];
      newDays[dayIndex] = { ...newDays[dayIndex - 1] };
      // Update day of week
      newDays[dayIndex].DayOfWeek = (newDays[dayIndex - 1].DayOfWeek + 1) % 7;
      // Update weekend flag
      newDays[dayIndex].Weekend = newDays[dayIndex].DayOfWeek >= 5 ? 1 : 0;
      setDays(newDays);
    }
  };

  const resetToDefaults = () => {
    setDays(Array(7).fill(null).map((_, i) => ({
      ...DEFAULT_DAY,
      DayOfWeek: i % 7,
      Weekend: (i % 7) >= 5 ? 1 : 0
    })));
    setPrediction(null);
    setError(null);
  };

  const dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

  return (
    <div className="test-panel">
      <h2>Model Test Panel</h2>
      <p className="panel-description">
        Enter 7 days of blood bank data to predict tomorrow's blood units needed.
        The model uses an LSTM neural network trained via Federated Learning.
      </p>

      <div className="day-tabs">
        {days.map((_, index) => (
          <button
            key={index}
            className={`day-tab ${activeDay === index ? 'active' : ''}`}
            onClick={() => setActiveDay(index)}
          >
            Day {index + 1}
          </button>
        ))}
      </div>

      <div className="day-input-form">
        <div className="day-header">
          <h3>Day {activeDay + 1} Data</h3>
          <div className="day-actions">
            {activeDay > 0 && (
              <button 
                className="copy-btn"
                onClick={() => copyFromPreviousDay(activeDay)}
              >
                Copy from Day {activeDay}
              </button>
            )}
          </div>
        </div>

        <div className="input-grid">
          <div className="input-group">
            <label>Day of Week</label>
            <select
              value={days[activeDay].DayOfWeek}
              onChange={(e) => handleInputChange(activeDay, 'DayOfWeek', e.target.value)}
            >
              {dayNames.map((name, i) => (
                <option key={i} value={i}>{name}</option>
              ))}
            </select>
          </div>

          <div className="input-group">
            <label>Month</label>
            <select
              value={days[activeDay].Month}
              onChange={(e) => handleInputChange(activeDay, 'Month', e.target.value)}
            >
              {Array.from({ length: 12 }, (_, i) => (
                <option key={i + 1} value={i + 1}>
                  {new Date(2000, i, 1).toLocaleString('default', { month: 'long' })}
                </option>
              ))}
            </select>
          </div>

          <div className="input-group">
            <label>Weekend</label>
            <select
              value={days[activeDay].Weekend}
              onChange={(e) => handleInputChange(activeDay, 'Weekend', e.target.value)}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>

          <div className="input-group">
            <label>Blood Type</label>
            <select
              value={days[activeDay].Blood_Type}
              onChange={(e) => handleInputChange(activeDay, 'Blood_Type', e.target.value)}
            >
              {BLOOD_TYPES.map(bt => (
                <option key={bt} value={bt}>{bt}</option>
              ))}
            </select>
          </div>

          <div className="input-group">
            <label>Emergency Room Cases</label>
            <input
              type="number"
              min="0"
              value={days[activeDay].Emergency_Room_Cases}
              onChange={(e) => handleInputChange(activeDay, 'Emergency_Room_Cases', e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>Scheduled Surgeries</label>
            <input
              type="number"
              min="0"
              value={days[activeDay].Scheduled_Surgeries}
              onChange={(e) => handleInputChange(activeDay, 'Scheduled_Surgeries', e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>Trauma Alert Level (0-5)</label>
            <input
              type="number"
              min="0"
              max="5"
              value={days[activeDay].Trauma_Alert_Level}
              onChange={(e) => handleInputChange(activeDay, 'Trauma_Alert_Level', e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>New Donations</label>
            <input
              type="number"
              min="0"
              value={days[activeDay].New_Donations}
              onChange={(e) => handleInputChange(activeDay, 'New_Donations', e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>Units Used Today</label>
            <input
              type="number"
              min="0"
              value={days[activeDay].Units_Used}
              onChange={(e) => handleInputChange(activeDay, 'Units_Used', e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>Starting Inventory</label>
            <input
              type="number"
              min="0"
              value={days[activeDay].Starting_Inventory}
              onChange={(e) => handleInputChange(activeDay, 'Starting_Inventory', e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>Ending Inventory</label>
            <input
              type="number"
              min="0"
              value={days[activeDay].Ending_Inventory}
              onChange={(e) => handleInputChange(activeDay, 'Ending_Inventory', e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>Days Supply</label>
            <input
              type="number"
              min="0"
              value={days[activeDay].Days_Supply}
              onChange={(e) => handleInputChange(activeDay, 'Days_Supply', e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>Shortage Flag</label>
            <select
              value={days[activeDay].Shortage_Flag}
              onChange={(e) => handleInputChange(activeDay, 'Shortage_Flag', e.target.value)}
            >
              <option value={0}>No Shortage</option>
              <option value={1}>Shortage</option>
            </select>
          </div>
        </div>
      </div>

      <div className="action-buttons">
        <button 
          className="predict-btn"
          onClick={handlePredict}
          disabled={loading}
        >
          {loading ? 'Predicting...' : '🔮 Predict Tomorrow\'s Usage'}
        </button>
        <button 
          className="reset-btn"
          onClick={resetToDefaults}
        >
          Reset to Defaults
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
            <span className="prediction-unit">blood units</span>
          </div>
          <p className="prediction-message">{prediction.message}</p>
        </div>
      )}

      <div className="data-summary">
        <h4>7-Day Data Summary</h4>
        <table className="summary-table">
          <thead>
            <tr>
              <th>Day</th>
              <th>ER Cases</th>
              <th>Surgeries</th>
              <th>Units Used</th>
              <th>Inventory</th>
              <th>Shortage</th>
            </tr>
          </thead>
          <tbody>
            {days.map((day, i) => (
              <tr key={i} className={activeDay === i ? 'active-row' : ''}>
                <td>Day {i + 1}</td>
                <td>{day.Emergency_Room_Cases}</td>
                <td>{day.Scheduled_Surgeries}</td>
                <td>{day.Units_Used}</td>
                <td>{day.Ending_Inventory}</td>
                <td>{day.Shortage_Flag ? '⚠️ Yes' : '✓ No'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default TestPanel;
