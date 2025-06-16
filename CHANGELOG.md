# Changelog

All notable changes to PHAL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Plugin marketplace integration
- Community recipe sharing via IPFS
- Advanced AI optimization algorithms
- Multi-language support framework

### Changed
- Improved VPD calculation accuracy
- Enhanced offline mode capabilities

### Fixed
- WebSocket reconnection issues
- Memory leak in sensor polling

## [3.0.0] - 2024-12-20

### Added
- Complete frontend dashboard with real-time updates
- Plugin architecture for extensibility
- AI assistant integration (Claude/GPT)
- Community features and recipe sharing
- Offline-first operation mode
- Advanced analytics and predictive modeling
- Multi-tenant support
- Comprehensive audit logging

### Changed
- Migrated to FastAPI from Flask
- Redesigned UI with modern component architecture
- Improved hardware abstraction layer
- Enhanced security with JWT authentication

### Deprecated
- Legacy XML configuration format
- v1 API endpoints (use v2)

### Removed
- PHP bridge (no longer needed)
- MySQL support (use PostgreSQL)

### Fixed
- Race condition in sensor readings
- Memory usage in long-running processes
- Cross-zone contamination in recipes

### Security
- Added rate limiting to all endpoints
- Implemented RBAC
- Enhanced input validation

## [2.0.0] - 2024-06-15

### Added
- Multi-zone support
- Recipe management system
- Automated nutrient dosing
- Harvest tracking
- Maintenance scheduling

### Changed
- Database schema for better performance
- API structure for RESTful compliance

## [1.0.0] - 2024-01-10

### Added
- Initial release
- Basic sensor monitoring
- Simple automation rules
- Data logging to InfluxDB
- Web dashboard

[Unreleased]: https://github.com/HydroFarmerJason/PHAL/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/HydroFarmerJason/PHAL/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/HydroFarmerJason/PHAL/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/HydroFarmerJason/PHAL/releases/tag/v1.0.0
