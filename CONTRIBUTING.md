# Contributing

Contributions are welcome — bug reports, feature requests, and pull requests.

## Setup

```bash
git clone https://github.com/MarouaneAEF/kalman_time_series.git
cd kalman_time_series
pip install -r requirements.txt
```

## Run tests

```bash
pip install pytest
pytest tests/ -v
```

## Run linter

```bash
pip install pylint
pylint $(git ls-files '*.py') --fail-under=7.0
```

## Submitting a PR

1. Fork the repo and create a branch from `main`
2. Make your changes and add tests if relevant
3. Ensure `pytest` and `pylint` pass
4. Open a pull request with a clear description
