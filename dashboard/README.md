# Federated Learning Dashboard

A modern React-based web dashboard for managing and monitoring Federated Learning training processes.

## Features

- **Server Management**: Start/stop the training server and configure parameters
- **Client Management**: Control multiple training clients with start/stop controls
- **Configuration**: Edit server configuration (host, port, number of clients, training rounds)
- **Real-time Monitoring**: View server and client status
- **Logs Viewer**: Real-time logs from server and all clients with auto-scroll and download capabilities

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Python 3.7+ with Flask (for API backend)

## Installation

### React Dashboard

```bash
cd dashboard
npm install
```

### Python API Backend

The dashboard communicates with Python API endpoints on the server and client machines. You need to modify your server and client code to expose these endpoints.

## Configuration

### Environment Variables

Create a `.env` file in the dashboard directory:

```env
REACT_APP_SERVER_URL=http://127.0.0.1:5000
REACT_APP_CLIENT_URL_BASE=http://127.0.0.1
```

**Note:** The client port will be calculated as `6000 + clientId` (so Client 1 runs on 6001, Client 2 on 6002, etc.)

## Running the Dashboard

```bash
npm start
```

The dashboard will open at `http://localhost:3000`

## API Endpoints

The dashboard expects the following API endpoints on your server and client:

### Server API (http://127.0.0.1:5000)

- `GET /config` - Get current server configuration
- `POST /config/update` - Update server configuration
- `POST /start` - Start the server
- `POST /stop` - Stop the server
- `GET /status` - Get server status
- `GET /logs` - Get server logs
- `POST /logs/clear` - Clear server logs

### Client API (http://127.0.0.1:6001+ )

- `GET /status` - Get client status
- `POST /start` - Start the client
- `POST /stop` - Stop the client
- `GET /logs` - Get client logs
- `POST /logs/clear` - Clear client logs

## Architecture

- **React Frontend**: Modern UI for dashboard management
- **API Service Layer**: Abstraction for server and client API calls
- **Components**:
  - `ServerControl`: Server management and configuration
  - `ClientControl`: Multi-client management interface
  - `LogsViewer`: Real-time logs display and management

## Server Setup

You need to add Flask-based API endpoints to your existing server. See `server_api.py` for reference implementation.

## Client Setup

You need to add Flask-based API endpoints to each client instance. See `client_api.py` for reference implementation.

## Technology Stack

- **React 18** - UI framework
- **Axios** - HTTP client
- **CSS3** - Styling with modern features

## License

MIT
