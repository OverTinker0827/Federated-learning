import React, { useState, useEffect } from 'react';
import { ServerAPI } from '../services/api';
import './ServerControl.css';

function ServerControl({ onConfigChange, onServerStatusChange, numClients }) {
  const [config, setConfig] = useState({
    host: '127.0.0.1',
    port: 5000,
    num_clients: 2,
    rounds: 1
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  const [serverRunning, setServerRunning] = useState(false);
  const [editMode, setEditMode] = useState(false);

  // Load config on mount
  useEffect(() => {
    loadConfig();
    checkServerStatus();
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
      [name]: name === 'num_clients' || name === 'port' || name === 'rounds'
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

      {message && (
        <div className={`message message-${messageType}`}>
          {message}
        </div>
      )}
    </div>
  );
}

export default ServerControl;
