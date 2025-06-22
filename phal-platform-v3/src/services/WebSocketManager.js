import EventEmitter from 'eventemitter3';
import AuthService from './AuthService';

export class WebSocketManager extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      url: config.url || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
      reconnectInterval: config.reconnectInterval || 1000,
      maxReconnectInterval: config.maxReconnectInterval || 30000,
      reconnectDecay: config.reconnectDecay || 1.5,
      maxReconnectAttempts: config.maxReconnectAttempts || null,
      enableHeartbeat: config.enableHeartbeat !== false,
      heartbeatInterval: config.heartbeatInterval || 30000
    };
    this.ws = null;
    this.reconnectAttempts = 0;
    this.reconnectTimeout = null;
    this.heartbeatInterval = null;
    this.messageQueue = [];
    this.isConnected = false;
    this.shouldReconnect = true;
  }

  connect() {
    return new Promise((resolve, reject) => {
      try {
        const token = AuthService.getToken();
        if (!token) {
          reject(new Error('No authentication token'));
          return;
        }
        this.ws = new WebSocket(this.config.url, token);
        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          if (this.config.enableHeartbeat) {
            this.startHeartbeat();
          }
          this.flushMessageQueue();
          this.emit('connected');
          resolve();
        };
        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            switch (data.type) {
              case 'pong':
                break;
              case 'event':
                this.emit(data.event, data.data);
                break;
              case 'error':
                this.emit('error', data);
                break;
              default:
                this.emit('message', data);
            }
          } catch (error) {
            console.error('WebSocket message parse error:', error);
          }
        };
        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.emit('error', error);
          reject(error);
        };
        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected');
          this.isConnected = false;
          this.stopHeartbeat();
          this.emit('disconnected', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean
          });
          if (this.shouldReconnect && !event.wasClean) {
            this.scheduleReconnect();
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  scheduleReconnect() {
    if (this.config.maxReconnectAttempts && this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.log('Max reconnection attempts reached');
      this.emit('reconnectFailed');
      return;
    }
    const timeout = Math.min(
      this.config.reconnectInterval * Math.pow(this.config.reconnectDecay, this.reconnectAttempts),
      this.config.maxReconnectInterval
    );
    console.log(`Scheduling reconnect in ${timeout}ms (attempt ${this.reconnectAttempts + 1})`);
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      this.emit('reconnecting', { attempt: this.reconnectAttempts });
      this.connect().catch(console.error);
    }, timeout);
  }

  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, this.config.heartbeatInterval);
  }

  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  send(data) {
    const message = typeof data === 'string' ? data : JSON.stringify(data);
    if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(message);
      return true;
    } else {
      this.messageQueue.push(message);
      return false;
    }
  }

  flushMessageQueue() {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      this.ws.send(message);
    }
  }

  disconnect() {
    this.shouldReconnect = false;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  reconnect() {
    this.shouldReconnect = true;
    this.disconnect();
    return this.connect();
  }
}

export default WebSocketManager;
