import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import ErrorBoundary from './components/ErrorBoundary';
import { setupMonitoring } from './utils/monitoring';
import './styles/index.css';

setupMonitoring();

const checkCompatibility = () => {
  const required = ['WebSocket', 'fetch', 'Promise', 'Map', 'Set'];
  const missing = required.filter(feature => !(feature in window));
  if (missing.length > 0) {
    console.error('Browser missing required features:', missing);
    document.body.innerHTML = `
      <div class="compatibility-error">
        <h1>Browser Not Supported</h1>
        <p>Please upgrade to a modern browser to use PHAL Platform.</p>
      </div>
    `;
    return false;
  }
  return true;
};

if (checkCompatibility()) {
  ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </React.StrictMode>
  );
}

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').then(
      (registration) => {
        console.log('SW registered:', registration);
      },
      (error) => {
        console.error('SW registration failed:', error);
      }
    );
  });
}
