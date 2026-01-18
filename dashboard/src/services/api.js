import axios from 'axios';
import clientConfig from '../client_config.json';

// Server API service
export const ServerAPI = {
  baseURL: process.env.REACT_APP_SERVER_URL || 'http://127.0.0.1:5000',

  async getConfig() {
    try {
      const response = await axios.get(`${this.baseURL}/config`);
      return response.data;
    } catch (error) {
      console.error('Error fetching server config:', error);
      throw error;
    }
  },

  async startServer(config) {
    try {
      const response = await axios.post(`${this.baseURL}/start`, config);
      return response.data;
    } catch (error) {
      console.error('Error starting server:', error);
      throw error;
    }
  },

  async stopServer() {
    try {
      const response = await axios.post(`${this.baseURL}/stop`);
      return response.data;
    } catch (error) {
      console.error('Error stopping server:', error);
      throw error;
    }
  },

  async getStatus() {
    try {
      const response = await axios.get(`${this.baseURL}/status`);
      return response.data;
    } catch (error) {
      console.error('Error fetching server status:', error);
      throw error;
    }
  },

  async getLogs() {
    try {
      const response = await axios.get(`${this.baseURL}/logs`);
      return response.data;
    } catch (error) {
      console.error('Error fetching server logs:', error);
      throw error;
    }
  },

  async clearLogs() {
    try {
      const response = await axios.post(`${this.baseURL}/logs/clear`);
      return response.data;
    } catch (error) {
      console.error('Error clearing server logs:', error);
      throw error;
    }
  },

  async updateConfig(config) {
    try {
      const response = await axios.post(`${this.baseURL}/config/update`, config);
      return response.data;
    } catch (error) {
      console.error('Error updating config:', error);
      throw error;
    }
  }
};

// Client API service
export const ClientAPI = {
  getClientConfig: (clientId) => {
    return clientConfig[`client${clientId}`] || null;
  },

  getBaseURL: (clientId) => {
    const clientCfg = clientConfig ? clientConfig[`client${clientId}`] : null;
    if (clientCfg && clientCfg.ip) return clientCfg.ip;
    return process.env.REACT_APP_CLIENT_URL_BASE || 'http://127.0.0.1';
  },

  getPort: (clientId) => {
    const clientCfg = clientConfig ? clientConfig[`client${clientId}`] : null;
    if (clientCfg && clientCfg.port) {
      // port might be a string in config; ensure we return as-is for URL
      return clientCfg.port;
    }
    return 6000 + Number(clientId); // Clients on ports 6001, 6002, etc.
  },

  async getClientURL(clientId) {
    let base = this.getBaseURL(clientId);
    const port = this.getPort(clientId);

    // Ensure base includes protocol
    if (!/^https?:\/\//i.test(base)) {
      base = `http://${base}`;
    }

    return `${base}:${port}`;
  },

  async getStatus(clientId) {
    try {
      const url = await this.getClientURL(clientId);
      const response = await axios.get(`${url}/status`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching client ${clientId} status:`, error);
      throw error;
    }
  },

  async startClient(clientId, config) {
    try {
      const url = await this.getClientURL(clientId);
      const response = await axios.post(`${url}/start`, config);
      return response.data;
    } catch (error) {
      console.error(`Error starting client ${clientId}:`, error);
      throw error;
    }
  },

  async stopClient(clientId) {
    try {
      const url = await this.getClientURL(clientId);
      const response = await axios.post(`${url}/stop`);
      return response.data;
    } catch (error) {
      console.error(`Error stopping client ${clientId}:`, error);
      throw error;
    }
  },

  async getLogs(clientId) {
    try {
      const url = await this.getClientURL(clientId);
      const response = await axios.get(`${url}/logs`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching client ${clientId} logs:`, error);
      throw error;
    }
  },

  async clearLogs(clientId) {
    try {
      const url = await this.getClientURL(clientId);
      const response = await axios.post(`${url}/logs/clear`);
      return response.data;
    } catch (error) {
      console.error(`Error clearing client ${clientId} logs:`, error);
      throw error;
    }
  }
};
