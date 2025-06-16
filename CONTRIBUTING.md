# Contributing to PHAL

First off, thank you for considering contributing to PHAL! It's people like you that make PHAL such a great tool for the agricultural community.

## Code of Conduct

This project and everyone participating in it is governed by the [PHAL Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots if possible
* Include your environment details (OS, Python version, browser, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful to most PHAL users

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the Python style guide (PEP 8)
* Include thoughtfully-worded, well-structured tests
* Document new code
* End all files with a newline

## Development Process

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/PHAL.git
cd phal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
flake8 backend/src/
black backend/src/ --check
```

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Style Guide

* Follow PEP 8
* Use type hints where possible
* Write docstrings for all public methods
* Keep functions focused and small
* Use meaningful variable names

### JavaScript Style Guide

* Use ES6+ features
* Use async/await over promises where possible
* Comment complex logic
* Keep components modular and reusable

## Growing Recipe Contributions

We especially welcome contributions of successful growing recipes! To contribute a recipe:

1. Use the recipe template in `examples/recipes/template.json`
2. Include detailed environmental parameters for each growth stage
3. Document any special considerations or tips
4. Include yield data if available
5. Add attribution and license information

## Plugin Development

See our [Plugin Development Guide](docs/plugin-development.md) for detailed information on creating PHAL plugins.

## Community

* [GitHub Discussions](https://github.com/HydroFarmerJason/PHAL/discussions)
* [GitHub Issues](https://github.com/HydroFarmerJason/PHAL/issues)

## Recognition

Contributors who submit accepted PRs will be added to our [Contributors list](CONTRIBUTORS.md) and receive recognition in the project.

Thank you for contributing to the future of open agriculture! 🌱
