import React, { useState, useEffect } from 'react';
import { ClientAPI } from '../services/api';
import './ClientControl.css';

function ClientControl({ numClients, serverRunning, serverConfig }) {
  const [clients, setClients] = useState([]);
  const [loading, setLoading] = useState({});
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');

  // Initialize clients
  useEffect(() => {
    const initializeClients = () => {
      const clientList = Array.from({ length: numClients }, (_, i) => ({
        id: i + 1,
        running: false,
        training: false,
        progress: 0,
        error: null
      }));
      setClients(clientList);
      // Check status of all clients
      clientList.forEach(client => checkClientStatus(client.id));
    };
    initializeClients();
  }, [numClients]);

  const checkClientStatus = async (clientId) => {
    try {
      const status = await ClientAPI.getStatus(clientId);
      setClients(prev => prev.map(c =>
        c.id === clientId
          ? { ...c, running: status.running, training: status.training, progress: status.progress || 0, error: null }
          : c
      ));
    } catch (error) {
      setClients(prev => prev.map(c =>
        c.id === clientId
          ? { ...c, running: false, error: 'Connection failed' }
          : c
      ));
    }
  };

  const showMessage = (text, type = 'success') => {
    setMessage(text);
    setMessageType(type);
    setTimeout(() => setMessage(''), 3000);
  };

  const handleStartClient = async (clientId) => {
    try {
      setLoading(prev => ({ ...prev, [clientId]: true }));
      const config = {
        server_ip: serverConfig?.server_ip || process.env.REACT_APP_SERVER_HOST || '172.16.0.5',
        server_port: serverConfig?.server_port || parseInt(process.env.REACT_APP_SERVER_PORT) || 5000,
        epochs: serverConfig?.epochs || 1
      };
      await ClientAPI.startClient(clientId, config);
      await checkClientStatus(clientId);
      showMessage(`Client ${clientId} started successfully!`, 'success');
    } catch (error) {
      showMessage(`Failed to start client ${clientId}: ${error.message}`, 'error');
    } finally {
      setLoading(prev => ({ ...prev, [clientId]: false }));
    }
  };

  const handleStopClient = async (clientId) => {
    try {
      setLoading(prev => ({ ...prev, [clientId]: true }));
      await ClientAPI.stopClient(clientId);
      await checkClientStatus(clientId);
      showMessage(`Client ${clientId} stopped successfully!`, 'success');
    } catch (error) {
      showMessage(`Failed to stop client ${clientId}: ${error.message}`, 'error');
    } finally {
      setLoading(prev => ({ ...prev, [clientId]: false }));
    }
  };

  return (
    <div className="client-control">
      <div className="clients-header">
        <h2>Client Management</h2>
        <p className="subtitle">Total Clients: {numClients}</p>
      </div>

      {!serverRunning && (
        <div className="warning-banner">
          ⚠️ Server is not running. Some features may not be available.
        </div>
      )}

      <div className="clients-grid">
        {clients.map(client => (
          <div key={client.id} className="client-card">
            <div className="card-header">
              <h3>Client {client.id}</h3>
              <div className={`status-dot ${client.running ? 'running' : 'stopped'}`} title={client.running ? 'Running' : 'Stopped'} />
            </div>

            <div className="card-body">
              <div className="info-row">
                <span className="label">Status:</span>
                <span className={`status-text ${client.running ? 'running' : 'stopped'}`}>
                  {client.running ? '● Running' : '○ Stopped'}
                </span>
              </div>

              {client.training && (
                <div className="info-row">
                  <span className="label">Training:</span>
                  <span className="training-indicator">In Progress ({client.progress}%)</span>
                </div>
              )}

              {client.error && (
                <div className="info-row error">
                  <span className="label">Error:</span>
                  <span className="error-text">{client.error}</span>
                </div>
              )}

              {client.training && (
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${client.progress}%` }}></div>
                </div>
              )}
            </div>

            <div className="card-footer">
              <button
                className="btn btn-success btn-small"
                onClick={() => handleStartClient(client.id)}
                disabled={client.running || loading[client.id] || !serverRunning}
              >
                {loading[client.id] ? '...' : 'Start'}
              </button>
              <button
                className="btn btn-danger btn-small"
                onClick={() => handleStopClient(client.id)}
                disabled={!client.running || loading[client.id]}
              >
                {loading[client.id] ? '...' : 'Stop'}
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="quick-actions">
        <button
          className="btn btn-primary"
          onClick={() => {
            clients.forEach(c => {
              if (!c.running && serverRunning) {
                handleStartClient(c.id);
              }
            });
          }}
          disabled={!serverRunning || clients.every(c => c.running)}
        >
          Start All
        </button>
        <button
          className="btn btn-danger"
          onClick={() => {
            clients.forEach(c => {
              if (c.running) {
                handleStopClient(c.id);
              }
            });
          }}
          disabled={!clients.some(c => c.running)}
        >
          Stop All
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => {
            clients.forEach(c => checkClientStatus(c.id));
          }}
        >
          Refresh Status
        </button>
      </div>

      {message && (
        <div className={`message message-${messageType}`}>
          {message}
        </div>
      )}
    </div>
  );
}

export default ClientControl;
