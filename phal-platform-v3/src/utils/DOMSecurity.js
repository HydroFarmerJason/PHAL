// Secure DOM manipulation utilities to prevent XSS
export class SecureDOM {
  static createElementFromTemplate(template, data) {
    const div = document.createElement('div');
    const sanitized = this.sanitizeData(data);
    const html = template(sanitized);
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    return doc.body.firstChild;
  }

  static sanitizeData(data) {
    if (typeof data === 'string') {
      return this.escapeHtml(data);
    }
    if (Array.isArray(data)) {
      return data.map(item => this.sanitizeData(item));
    }
    if (typeof data === 'object' && data !== null) {
      const sanitized = {};
      for (const [key, value] of Object.entries(data)) {
        sanitized[key] = this.sanitizeData(value);
      }
      return sanitized;
    }
    return data;
  }

  static escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  static setTextContent(element, text) {
    element.textContent = text;
  }

  static setAttribute(element, name, value) {
    if (!/^[a-zA-Z][\w\-:]*$/.test(name)) {
      throw new Error('Invalid attribute name');
    }
    if (['href', 'src', 'action'].includes(name.toLowerCase())) {
      value = this.sanitizeUrl(value);
    }
    element.setAttribute(name, value);
  }

  static sanitizeUrl(url) {
    const allowedProtocols = ['http:', 'https:', 'mailto:'];
    try {
      const parsed = new URL(url);
      if (!allowedProtocols.includes(parsed.protocol)) {
        return '#';
      }
      return url;
    } catch {
      return '#';
    }
  }

  static addEventListener(element, event, handler, options) {
    const secureHandler = (e) => {
      try {
        handler(e);
      } catch (error) {
        console.error('Event handler error:', error);
      }
    };
    element.addEventListener(event, secureHandler, options);
    return () => element.removeEventListener(event, secureHandler, options);
  }
}
