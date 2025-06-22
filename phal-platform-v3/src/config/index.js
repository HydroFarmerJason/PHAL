const config = {
  api: {
    url: import.meta.env.VITE_API_URL || 'http://localhost:8080',
    timeout: 30000,
    retries: 3
  },
  websocket: {
    url: import.meta.env.VITE_WS_URL || 'ws://localhost:8080/ws',
    reconnectInterval: 1000,
    maxReconnectInterval: 30000,
    heartbeatInterval: 30000
  },
  security: {
    csrfHeaderName: import.meta.env.VITE_CSRF_HEADER_NAME || 'X-CSRF-Token',
    sessionTimeout: parseInt(import.meta.env.VITE_SESSION_TIMEOUT) || 3600000
  },
  features: {
    telemetry: import.meta.env.VITE_ENABLE_TELEMETRY === 'true',
    offline: import.meta.env.VITE_ENABLE_OFFLINE_MODE === 'true',
    ai: import.meta.env.VITE_ENABLE_AI === 'true'
  },
  development: {
    mockApi: import.meta.env.VITE_MOCK_API === 'true',
    debug: import.meta.env.VITE_DEBUG_MODE === 'true'
  }
};

export default config;
