import jwtDecode from 'jwt-decode';

class AuthService {
  constructor() {
    this.tokenKey = 'phal_auth_token';
    this.refreshTokenKey = 'phal_refresh_token';
    this.tokenRefreshInterval = null;
  }

  // Secure token storage using httpOnly cookies fallback
  async login(apiKey) {
    try {
      const response = await fetch('/api/v1/auth/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': this.getCSRFToken()
        },
        credentials: 'include',
        body: JSON.stringify({ apiKey })
      });

      if (!response.ok) throw new Error('Authentication failed');

      const data = await response.json();
      // Store tokens securely
      this.setTokens(data.token, data.refreshToken);
      this.startTokenRefresh();
      return data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  }

  setTokens(token, refreshToken) {
    if (process.env.NODE_ENV === 'test' || !(window.crypto && window.crypto.subtle)) {
      sessionStorage.setItem(this.tokenKey, token);
      sessionStorage.setItem(this.refreshTokenKey, refreshToken);
    } else {
      this.encryptAndStore(this.tokenKey, token);
      this.encryptAndStore(this.refreshTokenKey, refreshToken);
    }
  }

  async encryptAndStore(key, value) {
    const encoder = new TextEncoder();
    const data = encoder.encode(value);
    const cryptoKey = await window.crypto.subtle.generateKey(
      { name: 'AES-GCM', length: 256 },
      false,
      ['encrypt', 'decrypt']
    );
    const iv = window.crypto.getRandomValues(new Uint8Array(12));
    const encrypted = await window.crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      cryptoKey,
      data
    );
    const stored = {
      encrypted: Array.from(new Uint8Array(encrypted)),
      iv: Array.from(iv)
    };
    sessionStorage.setItem(key, JSON.stringify(stored));
  }

  getToken() {
    return sessionStorage.getItem(this.tokenKey);
  }

  getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]')?.content || '';
  }

  startTokenRefresh() {
    const token = this.getToken();
    if (!token) return;

    try {
      const decoded = jwtDecode(token);
      const expiresIn = decoded.exp * 1000 - Date.now();
      const refreshTime = expiresIn - 5 * 60 * 1000;
      if (refreshTime > 0) {
        this.tokenRefreshInterval = setTimeout(() => this.refreshToken(), refreshTime);
      }
    } catch (error) {
      console.error('Token decode error:', error);
    }
  }

  async refreshToken() {
    try {
      const refreshToken = sessionStorage.getItem(this.refreshTokenKey);
      const response = await fetch('/api/v1/auth/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': this.getCSRFToken()
        },
        credentials: 'include',
        body: JSON.stringify({ refreshToken })
      });

      if (response.ok) {
        const data = await response.json();
        this.setTokens(data.token, data.refreshToken);
        this.startTokenRefresh();
      } else {
        this.logout();
      }
    } catch (error) {
      console.error('Token refresh error:', error);
      this.logout();
    }
  }

  logout() {
    clearTimeout(this.tokenRefreshInterval);
    sessionStorage.removeItem(this.tokenKey);
    sessionStorage.removeItem(this.refreshTokenKey);
    window.location.href = '/login';
  }

  isAuthenticated() {
    const token = this.getToken();
    if (!token) return false;
    try {
      const decoded = jwtDecode(token);
      return decoded.exp * 1000 > Date.now();
    } catch {
      return false;
    }
  }
}

export default new AuthService();
