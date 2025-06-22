#!/bin/bash
set -e

echo "🚀 Upgrading PHAL Backend..."

# Backup existing models
if [ -d "./models" ]; then
    echo "📦 Backing up ML models..."
    tar -czf models_backup_$(date +%Y%m%d_%H%M%S).tar.gz ./models/
fi

# Update dependencies
echo "📦 Installing dependencies..."
pip install --upgrade \
    aiohttp==3.9.5 \
    asyncpg==0.29.0 \
    redis[hiredis]==5.0.3 \
    pydantic==2.6.4 \
    joblib==1.3.2 \
    structlog==24.1.0 \
    aiohttp-cors==0.7.0 \
    prometheus-client==0.20.0 \
    opentelemetry-api==1.24.0 \
    opentelemetry-sdk==1.24.0 \
    opentelemetry-instrumentation-aiohttp-server==0.45b0 \
    opentelemetry-instrumentation-asyncpg==0.45b0

# Run database migrations
echo "🗄️  Running database migrations..."
python -c "
import asyncio
from phal_backend import PHALApplication

async def migrate():
    app = PHALApplication()
    await app.phal.initialize()
    print('✅ Database schema updated')
    await app.phal.shutdown()

asyncio.run(migrate())
"

echo "✅ PHAL Backend upgrade complete!"
