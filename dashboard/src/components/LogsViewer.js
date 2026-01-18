import React, { useState, useEffect, useRef } from 'react';
import { ServerAPI, ClientAPI } from '../services/api';
import './LogsViewer.css';

function LogsViewer({ numClients }) {
  const [selectedSource, setSelectedSource] = useState('server');
  const [selectedClient, setSelectedClient] = useState(1);
  const [logs, setLogs] = useState('');
  const [loading, setLoading] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(1000);
  const logsEndRef = useRef(null);

  // Auto-scroll effect
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  // Fetch logs
  const fetchLogs = async () => {
    try {
      setLoading(true);
      let logsData;
      if (selectedSource === 'server') {
        const response = await ServerAPI.getLogs();
        logsData = response.logs || response || '';
      } else {
        const response = await ClientAPI.getLogs(selectedClient);
        logsData = response.logs || response || '';
      }
      setLogs(typeof logsData === 'string' ? logsData : JSON.stringify(logsData, null, 2));
    } catch (error) {
      setLogs(`Error fetching logs: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchLogs();
  }, [selectedSource, selectedClient]);

  // Auto-refresh logs
  useEffect(() => {
    if (refreshInterval === 0) return;

    const interval = setInterval(() => {
      fetchLogs();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [selectedSource, selectedClient, refreshInterval]);

  const handleClearLogs = async () => {
    if (window.confirm('Are you sure you want to clear logs?')) {
      try {
        if (selectedSource === 'server') {
          await ServerAPI.clearLogs();
        } else {
          await ClientAPI.clearLogs(selectedClient);
        }
        setLogs('');
      } catch (error) {
        setLogs(`Error clearing logs: ${error.message}`);
      }
    }
  };

  const handleDownloadLogs = () => {
    const element = document.createElement('a');
    const file = new Blob([logs], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `${selectedSource}-${selectedSource === 'client' ? selectedClient : 'logs'}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleCopyLogs = () => {
    navigator.clipboard.writeText(logs);
  };

  return (
    <div className="logs-viewer">
      <div className="logs-controls">
        <div className="control-group">
          <label>Log Source</label>
          <div className="button-toggle">
            <button
              className={`toggle-btn ${selectedSource === 'server' ? 'active' : ''}`}
              onClick={() => setSelectedSource('server')}
            >
              Server
            </button>
            <button
              className={`toggle-btn ${selectedSource === 'client' ? 'active' : ''}`}
              onClick={() => setSelectedSource('client')}
            >
              Clients
            </button>
          </div>
        </div>

        {selectedSource === 'client' && (
          <div className="control-group">
            <label>Select Client</label>
            <select
              value={selectedClient}
              onChange={(e) => setSelectedClient(parseInt(e.target.value))}
              className="select-input"
            >
              {Array.from({ length: numClients }, (_, i) => (
                <option key={i + 1} value={i + 1}>
                  Client {i + 1}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="control-group">
          <label>Auto Refresh</label>
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
            className="select-input"
          >
            <option value={0}>Off</option>
            <option value={500}>500ms</option>
            <option value={1000}>1s</option>
            <option value={2000}>2s</option>
            <option value={5000}>5s</option>
          </select>
        </div>

        <div className="control-group checkbox">
          <label>
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            Auto Scroll
          </label>
        </div>
      </div>

      <div className="logs-container">
        <div className="logs-display">
          <pre>{logs || 'No logs available...'}</pre>
          <div ref={logsEndRef} />
        </div>
      </div>

      <div className="logs-actions">
        <button
          className="btn btn-primary"
          onClick={fetchLogs}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Refresh'}
        </button>
        <button
          className="btn btn-secondary"
          onClick={handleCopyLogs}
        >
          Copy to Clipboard
        </button>
        <button
          className="btn btn-secondary"
          onClick={handleDownloadLogs}
        >
          Download Logs
        </button>
        <button
          className="btn btn-danger"
          onClick={handleClearLogs}
        >
          Clear Logs
        </button>
      </div>
    </div>
  );
}

export default LogsViewer;
