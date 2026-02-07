import React, { useState, useEffect } from 'react';
import { ServerAPI } from '../services/api';
import './ServerControl.css';

function ServerControl({ onConfigChange, onServerStatusChange, numClients }) {
  const [config, setConfig] = useState({
    host: process.env.REACT_APP_SERVER_HOST || '20.212.89.239',
    port: parseInt(process.env.REACT_APP_SERVER_PORT) || 5000,
    num_clients: 2,
    rounds: 1,
    epochs: 1
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  const [serverRunning, setServerRunning] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(false);

  // Load config on mount and set up metrics polling
  useEffect(() => {
    loadConfig();
    checkServerStatus();
    
    // Poll for metrics every 2 seconds continuously
    const metricsInterval = setInterval(fetchMetrics, 2000);
    
    return () => {
      if (metricsInterval) clearInterval(metricsInterval);
    };
  }, []);

  const loadConfig = async () => {
    try {
      const data = await ServerAPI.getConfig();
      setConfig(data);
    } catch (error) {
      showMessage('Could not load config. Server may not be running yet.', 'info');
    }
  };

  const checkServerStatus = async () => {
    try {
      const status = await ServerAPI.getStatus();
      setServerRunning(status.running || false);
    } catch (error) {
      setServerRunning(false);
    }
  };

  const showMessage = (text, type = 'success') => {
    setMessage(text);
    setMessageType(type);
    setTimeout(() => setMessage(''), 3000);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: name === 'num_clients' || name === 'port' || name === 'rounds' || name === 'epochs'
        ? parseInt(value)
        : value
    }));
  };

  const handleSaveConfig = async () => {
    try {
      setLoading(true);
      await ServerAPI.updateConfig(config);
      onConfigChange(config);
      showMessage('Configuration saved successfully!', 'success');
      setEditMode(false);
    } catch (error) {
      showMessage('Failed to save configuration', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleStartServer = async () => {
    try {
      setLoading(true);
      await ServerAPI.startServer(config);
      setServerRunning(true);
      onServerStatusChange(true);
      showMessage('Server started successfully!', 'success');
    } catch (error) {
      showMessage(`Failed to start server: ${error.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleStopServer = async () => {
    try {
      setLoading(true);
      await ServerAPI.stopServer();
      setServerRunning(false);
      onServerStatusChange(false);
      showMessage('Server stopped successfully!', 'success');
    } catch (error) {
      showMessage(`Failed to stop server: ${error.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchMetrics = async () => {
    try {
      setMetricsLoading(true);
      const data = await ServerAPI.getMetrics();
      setMetrics(data.metrics || null);
    } catch (error) {
      // Silently fail if metrics not available yet
      setMetrics(null);
    } finally {
      setMetricsLoading(false);
    }
  };

  return (
    <div className="server-control">
      <div className="control-section">
        <h2>Server Configuration</h2>

        <div className="config-panel">
          <div className="config-row">
            <label>Host</label>
            <input
              type="text"
              name="host"
              value={config.host}
              onChange={handleInputChange}
              disabled={!editMode || serverRunning}
              className="config-input"
            />
          </div>

          <div className="config-row">
            <label>Port</label>
            <input
              type="number"
              name="port"
              value={config.port}
              onChange={handleInputChange}
              disabled={!editMode || serverRunning}
              className="config-input"
            />
          </div>

          <div className="config-row">
            <label>Number of Clients</label>
            <input
              type="number"
              name="num_clients"
              value={config.num_clients}
              onChange={handleInputChange}
              disabled={!editMode || serverRunning}
              min="1"
              max="10"
              className="config-input"
            />
          </div>

          <div className="config-row">
            <label>Training Rounds</label>
            <input
              type="number"
              name="rounds"
              value={config.rounds}
              onChange={handleInputChange}
              disabled={!editMode || serverRunning}
              min="1"
              max="100"
              className="config-input"
            />
          </div>

          <div className="config-row">
            <label>Epochs per Round</label>
            <input
              type="number"
              name="epochs"
              value={config.epochs}
              onChange={handleInputChange}
              disabled={!editMode || serverRunning}
              min="1"
              max="100"
              className="config-input"
            />
          </div>
        </div>

        <div className="button-group">
          {!editMode ? (
            <button
              className="btn btn-primary"
              onClick={() => setEditMode(true)}
              disabled={serverRunning}
            >
              Edit Configuration
            </button>
          ) : (
            <>
              <button
                className="btn btn-success"
                onClick={handleSaveConfig}
                disabled={loading}
              >
                {loading ? 'Saving...' : 'Save Configuration'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setEditMode(false);
                  loadConfig();
                }}
                disabled={loading}
              >
                Cancel
              </button>
            </>
          )}
        </div>
      </div>

      <div className="control-section">
        <h2>Server Control</h2>

        <div className="server-status">
          <div className={`status-badge ${serverRunning ? 'running' : 'stopped'}`}>
            {serverRunning ? '● Running' : '○ Stopped'}
          </div>
        </div>

        <div className="button-group">
          <button
            className="btn btn-success btn-large"
            onClick={handleStartServer}
            disabled={serverRunning || loading}
          >
            {loading ? 'Starting...' : 'Start Server'}
          </button>
          <button
            className="btn btn-danger btn-large"
            onClick={handleStopServer}
            disabled={!serverRunning || loading}
          >
            {loading ? 'Stopping...' : 'Stop Server'}
          </button>
        </div>
      </div>

      {metrics && (
        <div className="control-section">
          <h2>Test Results (Model Evaluation Metrics)</h2>
          <div className="metrics-panel">
            <div className="metric-row">
              <span className="metric-label">MSE (Mean Squared Error):</span>
              <span className="metric-value">{metrics.mse.toFixed(4)}</span>
            </div>
            <div className="metric-row">
              <span className="metric-label">MAE (Mean Absolute Error):</span>
              <span className="metric-value">{metrics.mae.toFixed(4)}</span>
            </div>
            <div className="metric-row">
              <span className="metric-label">RMSE (Root Mean Squared Error):</span>
              <span className="metric-value">{metrics.rmse.toFixed(4)}</span>
            </div>
            <div className="metric-row">
              <span className="metric-label">R² Score:</span>
              <span className="metric-value">{metrics.r2.toFixed(4)}</span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Samples Tested:</span>
              <span className="metric-value">{metrics.num_samples}</span>
            </div>
            {metrics.feature_dim && (
              <div className="metric-row">
                <span className="metric-label">Features:</span>
                <span className="metric-value">{metrics.feature_dim}</span>
              </div>
            )}
            {metrics.seq_len && (
              <div className="metric-row">
                <span className="metric-label">Sequence Length:</span>
                <span className="metric-value">{metrics.seq_len} days</span>
              </div>
            )}
          </div>

          {metrics.prediction_stats && (
            <div className="prediction-stats">
              <h3>Prediction Statistics</h3>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-label">Actual Range:</span>
                  <span className="stat-value">
                    {metrics.prediction_stats.y_true_min.toFixed(2)} - {metrics.prediction_stats.y_true_max.toFixed(2)}
                  </span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Predicted Range:</span>
                  <span className="stat-value">
                    {metrics.prediction_stats.y_pred_min.toFixed(2)} - {metrics.prediction_stats.y_pred_max.toFixed(2)}
                  </span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Actual Mean:</span>
                  <span className="stat-value">{metrics.prediction_stats.y_true_mean.toFixed(2)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Predicted Mean:</span>
                  <span className="stat-value">{metrics.prediction_stats.y_pred_mean.toFixed(2)}</span>
                </div>
              </div>
            </div>
          )}

          {metrics.sample_predictions && metrics.sample_predictions.length > 0 && (
            <div className="sample-predictions">
              <h3>Sample Predictions (Units Used Tomorrow)</h3>
              <div className="predictions-table-container">
                <table className="predictions-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Actual</th>
                      <th>Predicted</th>
                      <th>Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.sample_predictions.map((pred, idx) => (
                      <tr key={idx} className={pred.error > 10 ? 'high-error' : ''}>
                        <td>{idx + 1}</td>
                        <td>{pred.actual}</td>
                        <td>{pred.predicted}</td>
                        <td className={pred.error > 10 ? 'error-high' : 'error-low'}>{pred.error}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {message && (
        <div className={`message message-${messageType}`}>
          {message}
        </div>
      )}
    </div>
  );
}

export default ServerControl;
