#!/bin/bash
# Generate self-signed certificates for development

echo "Generating self-signed certificates for development..."

# Create certs directory if it doesn't exist
mkdir -p certs

# Generate private key
openssl genrsa -out certs/server.key 2048

# Generate certificate signing request
openssl req -new -key certs/server.key -out certs/server.csr -subj "/C=US/ST=State/L=City/O=PHAL Development/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -days 365 -in certs/server.csr -signkey certs/server.key -out certs/server.crt

# Clean up CSR
rm certs/server.csr

# Set appropriate permissions
chmod 600 certs/server.key
chmod 644 certs/server.crt

echo "✅ Certificates generated successfully!"
echo "   - certs/server.key (private key)"
echo "   - certs/server.crt (certificate)"
echo ""
echo "⚠️  These are self-signed certificates for DEVELOPMENT ONLY"
echo "   For production, use certificates from a trusted CA"
