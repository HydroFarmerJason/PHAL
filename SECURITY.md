# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 3.0.x   | :white_check_mark: |
| 2.0.x   | :white_check_mark: |
| 1.0.x   | :x:                |

## Reporting a Vulnerability

The PHAL team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

To report a security vulnerability, please use the GitHub Security Advisory ["Report a Vulnerability"](https://github.com/HydroFarmerJason/PHAL/security/advisories/new) tab.

The PHAL team will send a response indicating the next steps in handling your report. After the initial reply to your report, we will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

Report security bugs in third-party modules to the person or team maintaining the module.

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine the affected versions
2. Audit code to find any potential similar problems
3. Prepare fixes for all releases still under maintenance
4. Release new security fix versions

## Security Best Practices

When deploying PHAL:

### API Security
- Always use HTTPS in production
- Rotate API keys regularly
- Use strong JWT secrets (32+ characters)
- Enable rate limiting
- Implement proper CORS policies

### Database Security
- Use strong passwords for all databases
- Enable SSL/TLS for database connections
- Regular backups with encryption
- Restrict database access by IP

### Hardware Security
- Isolate control systems from public networks
- Use VPNs for remote access
- Regular firmware updates
- Physical security for critical hardware

### Data Protection
- Encrypt sensitive data at rest
- Use secure communication protocols (MQTTS, HTTPS)
- Regular security audits
- Comply with local data protection regulations

## Security Features

PHAL includes several security features:

- JWT-based authentication
- Role-based access control (RBAC)
- Audit logging of all system actions
- Encrypted storage for sensitive configuration
- Input validation and sanitization
- SQL injection protection
- XSS prevention
- CSRF protection

## Dependencies

We regularly update dependencies to patch known vulnerabilities. To check for vulnerabilities in your installation:

```bash
# Python dependencies
pip install safety
safety check

# Node dependencies
npm audit
```

## Contact

For any security concerns that should not be public, please use GitHub's security advisory feature or contact the maintainers directly through GitHub.

Thank you for helping keep PHAL and our users safe!
