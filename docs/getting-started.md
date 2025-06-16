# Getting Started with PHAL

Welcome to PHAL! This guide will help you get up and running quickly.

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (optional but recommended)
- Git

## Quick Start (Simulator Mode)

The fastest way to try PHAL is using simulator mode - no hardware required!

1. **Clone the repository**
   ```bash
   git clone https://github.com/HydroFarmerJason/PHAL.git
   cd phal
   ```

2. **Start the frontend**
   ```bash
   cd frontend
   python -m http.server 8000
   ```

3. **Open your browser**
   Navigate to `http://localhost:8000`

You'll see the PHAL dashboard with simulated sensors and zones. Try:
- Clicking different zones
- Adjusting environmental controls
- Viewing analytics
- Creating a recipe

## Full Installation

For production use with real hardware:

1. **Run the setup script**
   ```bash
   ./scripts/setup.sh
   ```

2. **Configure your environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   nano .env
   ```

3. **Start with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access the dashboard**
   Open `http://localhost:8080` in your browser

## Hardware Setup

### Supported Sensors

PHAL supports a wide range of sensors:

- **Temperature/Humidity**: DHT22, SHT31, BME280
- **pH**: Atlas Scientific, DFRobot
- **EC**: Atlas Scientific, DFRobot
- **CO2**: MH-Z19, K30
- **Light**: TSL2561, VEML7700

### Wiring Guide

See our [Hardware Compatibility Guide](hardware-compatibility.md) for detailed wiring diagrams.

### Basic Arduino Setup

```cpp
// Example Arduino sketch for PHAL
#include <ArduinoJson.h>

void setup() {
  Serial.begin(115200);
}

void loop() {
  StaticJsonDocument<200> doc;
  
  // Read sensors
  doc["temperature"] = readTemperature();
  doc["humidity"] = readHumidity();
  doc["timestamp"] = millis();
  
  // Send to PHAL
  serializeJson(doc, Serial);
  Serial.println();
  
  delay(5000); // 5 second interval
}
```

## Creating Your First Zone

1. **Access Zone Manager**
   Click the gear icon next to "Production Zones"

2. **Create New Zone**
   - Name: "Lettuce Nursery"
   - Type: "Nursery"
   - Units: "A1,A2,A3,A4"

3. **Set Environmental Targets**
   - Temperature: 22°C
   - Humidity: 65%
   - Photoperiod: 18 hours

4. **Add Sensors**
   Configure your hardware addresses in the zone settings

## Your First Recipe

1. **Open Recipe Builder**
   Click "Recipe Builder" in the quick actions

2. **Create Recipe**
   - Name: "Buttercrunch Lettuce"
   - Crop Type: "Lettuce"

3. **Add Growth Stages**
   - Germination (7 days)
   - Seedling (14 days)
   - Vegetative (21 days)
   - Harvest

4. **Save and Apply**
   Apply the recipe to your zone

## Monitoring Your Grow

- **Real-time Data**: See live sensor readings
- **Analytics**: View trends and patterns
- **Alarms**: Get notified of issues
- **AI Insights**: Receive optimization suggestions

## Next Steps

- [Configure Hardware](hardware-compatibility.md)
- [Develop Plugins](plugin-development.md)
- [Join the Community](https://discord.gg/phal)
- [Read API Docs](api-reference.md)

## Troubleshooting

### Common Issues

**Can't connect to dashboard**
- Check if all services are running: `docker-compose ps`
- Verify no firewall blocking ports 8080, 8086

**Sensors not reading**
- Check wiring connections
- Verify I2C is enabled (Raspberry Pi)
- Test with diagnostic script: `python scripts/test-sensors.py`

**Database errors**
- Check PostgreSQL is running
- Verify credentials in .env
- Run migrations: `docker-compose exec phal-api alembic upgrade head`

### Getting Help

- GitHub Issues: Bug reports and feature requests
- Discord: Real-time community support
- Forum: Detailed discussions
- Email: support@phal.io

Welcome to the PHAL community! 🌱
