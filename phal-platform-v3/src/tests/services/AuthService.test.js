import { describe, it, expect, beforeEach, vi } from 'vitest';
import AuthService from '@/services/AuthService';

describe('AuthService', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.clearAllMocks();
  });

  describe('login', () => {
    it('should authenticate with valid API key', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ token: 'mock-token', refreshToken: 'mock-refresh-token' })
      });
      const result = await AuthService.login('test-api-key');
      expect(result).toHaveProperty('token');
      expect(sessionStorage.getItem('phal_auth_token')).toBeTruthy();
    });

    it('should handle authentication failure', async () => {
      global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 401 });
      await expect(AuthService.login('invalid-key')).rejects.toThrow('Authentication failed');
    });
  });

  describe('token management', () => {
    it('should check if authenticated', () => {
      const validToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjk5OTk5OTk5OTl9.K_lUwtGbvjCHP8Ff-gW9GykydkkXl-RLjmJVWgqFiVU';
      sessionStorage.setItem('phal_auth_token', validToken);
      expect(AuthService.isAuthenticated()).toBe(true);
    });

    it('should detect expired token', () => {
      const expiredToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjB9.m3HGIQ8L9LcVDFqTyParser25YTYnGBoHDwXiYr84DE';
      sessionStorage.setItem('phal_auth_token', expiredToken);
      expect(AuthService.isAuthenticated()).toBe(false);
    });
  });
});
