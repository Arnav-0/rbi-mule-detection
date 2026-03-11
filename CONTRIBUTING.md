# Contributing to RBI Mule Detection

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork** the repository and clone your fork
2. Create a virtual environment: `python -m venv .venv`
3. Install in editable mode: `pip install -e .`
4. Run the tests: `make test`
5. Create a feature branch: `git checkout -b feature/your-feature`

## Development Workflow

### Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting
- Line length limit: **100 characters**
- Target Python version: **3.10+**
- Run `make lint` before committing

### Testing

- Write tests for all new features in the `tests/` directory
- Tests use `pytest` — run with `make test`
- Use `@pytest.mark.unit` for fast unit tests
- Use `@pytest.mark.integration` for tests requiring real data

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add temporal burst detection feature
fix: correct NaN handling in velocity features
docs: update README with setup instructions
test: add unit tests for graph network features
refactor: simplify feature pipeline orchestration
```

## Project Structure

```
src/
├── data/           # Data loading, merging, preprocessing, splitting
├── features/       # Feature engineering (57 features in 8 groups)
├── models/         # Model wrappers, training, evaluation, calibration
├── explainability/ # SHAP, fairness, model cards, NL explanations
├── temporal/       # Suspicious window detection
├── api/            # FastAPI REST API
├── db/             # Database utilities
└── utils/          # Config, constants, logging, metrics
```

## Adding a New Feature

1. Create a class in `src/features/` extending `BaseFeatureGenerator`
2. Implement the `compute()` method returning a DataFrame indexed by `account_id`
3. Register features in `src/features/registry.py`
4. Add the generator to the pipeline in `src/features/pipeline.py`
5. Write unit tests in `tests/test_features/`

## Adding a New Model

1. Create a wrapper in `src/models/` extending `BaseModelWrapper`
2. Implement `get_optuna_params()` and `build_model()`
3. Add the model name to `MODEL_NAMES` in `src/utils/constants.py`
4. Write tests in `tests/test_models/`

## Pull Request Process

1. Ensure all tests pass: `make test`
2. Ensure code passes lint: `make lint`
3. Update documentation if adding new features
4. Open a PR with a clear description of changes
5. Request review from a maintainer

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include: steps to reproduce, expected vs actual behavior, environment details
- For security vulnerabilities, email the maintainers directly — do not open a public issue

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
