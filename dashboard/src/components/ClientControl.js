import React, { useState, useEffect } from 'react';
import { ClientAPI } from '../services/api';
import './ClientControl.css';

const PHASE_TEXT = {
  connecting: 'Connecting to server',
  waiting_for_round: 'Waiting for round',
  weights_ready: 'Waiting for phone to download',
  phone_training: 'Training on phone',
  uploading: 'Uploading update',
  aggregating: 'Server aggregating',
  complete: 'Complete',
  error: 'Error',
};

const isPhone = (id) => ClientAPI.getClientConfig(id)?.type === 'phone';

function ClientControl({ numClients, serverRunning, serverConfig }) {
  const [clients, setClients] = useState([]);
  const [loading, setLoading] = useState({});
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');

  // Poll status so the phone's live progress shows up without clicking Refresh
  useEffect(() => {
    const t = setInterval(() => {
      for (let id = 1; id <= numClients; id++) checkClientStatus(id);
    }, 2000);
    return () => clearInterval(t);
  }, [numClients]);

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
          ? {
              ...c, running: status.running, training: status.training, progress: status.progress || 0, error: null,
              phase: status.phase, round: status.round, phoneOnline: status.phone_online, device: status.device
            }
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
        // address/port of the raw TCP FL server, as seen from the client processes on this laptop
        server_ip: process.env.REACT_APP_SERVER_HOST || '127.0.0.1',
        server_port: parseInt(process.env.REACT_APP_FL_PORT, 10) || serverConfig?.fl_port || 8765,
        fl_port: parseInt(process.env.REACT_APP_FL_PORT, 10) || serverConfig?.fl_port || 8765,
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
          <div key={client.id} className={`client-card ${isPhone(client.id) ? 'phone-card' : ''}`}>
            <div className="card-header">
              <h3>
                {isPhone(client.id) ? '📱 ' : '💻 '}Client {client.id}
                <span className="client-label">{ClientAPI.getClientConfig(client.id)?.label || ''}</span>
              </h3>
              <div className={`status-dot ${client.running ? 'running' : 'stopped'}`} title={client.running ? 'Running' : 'Stopped'} />
            </div>

            <div className="card-body">
              <div className="info-row">
                <span className="label">Status:</span>
                <span className={`status-text ${client.running ? 'running' : 'stopped'}`}>
                  {client.running ? '● Running' : '○ Stopped'}
                </span>
              </div>

              {isPhone(client.id) && (
                <>
                  <div className="info-row">
                    <span className="label">Phone app:</span>
                    <span className={`status-text ${client.phoneOnline ? 'running' : 'stopped'}`}>
                      {client.phoneOnline ? `● Online${client.device?.name ? ` (${client.device.name})` : ''}` : '○ Not connected'}
                    </span>
                  </div>
                  {client.running && (
                    <div className="info-row">
                      <span className="label">Phase:</span>
                      <span>{PHASE_TEXT[client.phase] || client.phase}{client.round ? ` · round ${client.round}` : ''}</span>
                    </div>
                  )}
                  {client.device?.battery != null && (
                    <div className="info-row">
                      <span className="label">Battery:</span>
                      <span>{client.device.battery}%{client.device.charging ? ' ⚡' : ''} · {client.device.cores || '?'} cores</span>
                    </div>
                  )}
                </>
              )}

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
