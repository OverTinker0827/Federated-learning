import React, { useState, useEffect } from 'react';
import './App.css';
import ServerControl from './components/ServerControl';
import ClientControl from './components/ClientControl';
import LogsViewer from './components/LogsViewer';
import { ServerAPI } from './services/api';

function App() {
  const [numClients, setNumClients] = useState(2);
  const [serverRunning, setServerRunning] = useState(false);
  const [activeTab, setActiveTab] = useState('server');
  const [loading, setLoading] = useState(false);

  // Load initial config
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const config = await ServerAPI.getConfig();
        setNumClients(config.num_clients || 2);
      } catch (error) {
        console.log('Could not load config - server may not be running');
      }
    };
    loadConfig();
  }, []);

  // Check server status periodically
  useEffect(() => {
    const checkStatus = setInterval(async () => {
      try {
        const status = await ServerAPI.getStatus();
        setServerRunning(status.running || false);
      } catch (error) {
        setServerRunning(false);
      }
    }, 2000);

    return () => clearInterval(checkStatus);
  }, []);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Federated Learning Dashboard</h1>
        <div className="header-info">
          <span className={`status-indicator ${serverRunning ? 'running' : 'stopped'}`}>
            {serverRunning ? '● Server Running' : '○ Server Stopped'}
          </span>
        </div>
      </header>

      <div className="app-content">
        <nav className="tab-navigation">
          <button
            className={`tab-btn ${activeTab === 'server' ? 'active' : ''}`}
            onClick={() => setActiveTab('server')}
          >
            Server Control
          </button>
          <button
            className={`tab-btn ${activeTab === 'clients' ? 'active' : ''}`}
            onClick={() => setActiveTab('clients')}
          >
            Clients ({numClients})
          </button>
          <button
            className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logs')}
          >
            Logs
          </button>
        </nav>

        <main className="tab-content">
          {activeTab === 'server' && (
            <ServerControl
              onConfigChange={(config) => setNumClients(config.num_clients)}
              onServerStatusChange={setServerRunning}
              numClients={numClients}
            />
          )}
          {activeTab === 'clients' && (
            <ClientControl numClients={numClients} serverRunning={serverRunning} />
          )}
          {activeTab === 'logs' && <LogsViewer numClients={numClients} />}
        </main>
      </div>
    </div>
  );
}

export default App;
